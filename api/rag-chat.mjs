// api/rag-chat.mjs (app-only, safe)
import { GoogleGenerativeAI } from '@google/generative-ai';
import { createClient } from '@supabase/supabase-js';
import crypto from 'crypto';

/* ---------- Env & Clients ---------- */
const FRONTEND_APP_KEY = process.env.FRONTEND_APP_KEY;            // REQUIRED
const MAX_REQ_PER_IP_DAY = Number(process.env.MAX_REQ_PER_IP_DAY || 200);
const MAX_MSG_LEN = Number(process.env.MAX_MSG_LEN || 1200);

if (!process.env.GOOGLE_API_KEY) throw new Error('missing GOOGLE_API_KEY');
if (!process.env.SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE) {
  throw new Error('missing Supabase env');
}
if (!FRONTEND_APP_KEY) throw new Error('missing FRONTEND_APP_KEY');

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const embedModel = genAI.getGenerativeModel({ model: 'text-embedding-004' });
// keep chat model, but we only answer from app data
const CHAT_MODEL = process.env.CHAT_MODEL || 'gemini-1.5-flash';
const PRIMARY = genAI.getGenerativeModel({ model: CHAT_MODEL });

const sb = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE);

/* ---------- CORS (tighten origin in prod) ---------- */
function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*'); // replace * with your app domain in prod
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-APP-KEY');
}

/* ---------- Tiny rate limit ---------- */
async function enforceRateLimit(req) {
  const ip = (req.headers['x-forwarded-for'] || req.socket?.remoteAddress || '').split(',')[0].trim();
  const day = new Date().toISOString().slice(0,10);
  await sb.from('request_log').insert({ day, ip }).select(); // best-effort
  const { count } = await sb.from('request_log')
    .select('*', { head: true, count: 'exact' })
    .eq('day', day).eq('ip', ip);
  if ((count || 0) > MAX_REQ_PER_IP_DAY) {
    const e = new Error('rate_limited_ip'); e.code = 429; throw e;
  }
}

/* ---------- Embedding cache ---------- */
async function getCachedEmbedding(text) {
  const id = crypto.createHash('sha256').update('GOOGLE:' + text).digest('hex');
  const { data } = await sb.from('embeddings_cache').select('vector').eq('id', id).single();
  return { id, vec: data?.vector || null };
}
async function putCachedEmbedding(id, vec) {
  await sb.from('embeddings_cache').upsert({ id, vector: vec }, { onConflict: 'id' });
}
async function embedOnce(text) {
  const { id, vec } = await getCachedEmbedding(text);
  if (vec) return vec;
  const er = await embedModel.embedContent({ content: { parts: [{ text }] } });
  const v = er.embedding.values;
  await putCachedEmbedding(id, v);
  return v;
}

/* ---------- System style (short) ---------- */
const systemStyle = (lang) => `
You are the in-app assistant for “Migrant-e-s Help” (Morocco).
Answer ONLY from app data (services, cities/categories, news, CAN 2025).
Style: concise, ${lang}, bullet points, include names/phones/addresses when relevant.
If info is missing, say you don't have it. Do not invent.
`.trim();

/* ---------- HTTP handler ---------- */
export default async function handler(req, res) {
  cors(res);
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'method_not_allowed' });

  // Require app key
  const gotKey = req.headers['x-app-key'];
  if (gotKey !== FRONTEND_APP_KEY) return res.status(401).json({ error: 'unauthorized' });

  try {
    await enforceRateLimit(req);

    const { chat_id, language = 'en', messages = [], filters = {} } = req.body || {};
    const lastUserRaw = [...messages].reverse().find(m => m.role === 'user')?.content || '';
    const lastUser = String(lastUserRaw).trim().slice(0, MAX_MSG_LEN);
    if (!lastUser) return res.status(400).json({ error: 'missing_user_message' });

    if (chat_id) await sb.from('messages').insert({ chat_id, role: 'user', content: lastUser });

    // 1) Embed (cached)
    const qvec = await embedOnce(lastUser);

    // 2) Retrieve from Supabase (vector RPCs) — tune counts as you like
    const [services, news, stadiums, places] = await Promise.all([
      sb.rpc('match_services',     { query_embedding: qvec, match_count: 6, city_id: filters.cityId ?? null, category_id: filters.categoryId ?? null }),
      sb.rpc('match_news',         { query_embedding: qvec, match_count: 3, category_id: filters.categoryId ?? null }),
      sb.rpc('match_can_stadiums', { query_embedding: qvec, match_count: 3, city_id: filters.cityId ?? null }),
      sb.rpc('match_places_can',   { query_embedding: qvec, match_count: 4, city_id: filters.cityId ?? null }),
    ]);

    // Log RPC errors (don’t fail the whole request)
    if (services.error)  console.error('services rpc error', services.error);
    if (news.error)      console.error('news rpc error', news.error);
    if (stadiums.error)  console.error('stadiums rpc error', stadiums.error);
    if (places.error)    console.error('places rpc error', places.error);

    // 3) Build short context for the model
    const trim = (s, n=300) => (s || '').toString().slice(0, n);
    const sec = (title, lines) => lines.length ? `\n${title}:\n${lines.join('\n')}` : '';

    const s1 = (services.data ?? []).map(h => `- ${trim(h.name_fr ?? h.name_en ?? h.name_ar ?? 'Service',80)} @ ${trim(h.address ?? '',80)} ${h.phone ? `(phone: ${trim(h.phone,30)})` : ''}`);
    const s2 = (places.data ?? []).map(h => `- ${trim(h.name_fr ?? h.name_en ?? h.name_ar ?? '',80)} (tourism)`);
    const s3 = (stadiums.data ?? []).map(h => `- ${trim(h.name_fr ?? h.name_en ?? h.name_ar ?? '',80)} stadium${h.capacity ? `, capacity: ${trim(h.capacity,20)}` : ''}`);
    const s4 = (news.data ?? []).map(h => `- ${trim(h.title_fr ?? h.title_en ?? h.title_ar ?? '',100)}`);

    const context = sec('Services', s1) + sec('Places to visit', s2) + sec('CAN 2025 Stadiums', s3) + sec('News', s4);

    const history = [
      { role: 'user', parts: [{ text: systemStyle(language) }] },
      { role: 'user', parts: [{ text: 'Context:\n' + (context || 'None') }] },
      // Only last 2 user/assistant turns; clamp size
      ...messages.slice(-2).map(m => ({ role: m.role, parts: [{ text: String(m.content || '').slice(0, MAX_MSG_LEN) }]})),
      // Finally, the user question is implicitly “the last message”
    ];

    // 4) Ask model (no aggressive retry)
    let output = '...';
    try {
      const chat = await PRIMARY.startChat({ history });
      const result = await chat.sendMessage('Answer the last user message from the provided context ONLY.');
      output = result.response?.text() ?? '...';
    } catch (e) {
      if (e?.status === 429) output = '⚠️ Temporarily rate-limited. Please try again in a moment.';
      else throw e;
    }

    if (chat_id) await sb.from('messages').insert({ chat_id, role: 'assistant', content: output });

    const sources = [
      ...(services.data ?? []).map(h => ({ type: 'service',  id: h.id, name: h.name_fr ?? h.name_en ?? h.name_ar, lat: h.lat, lng: h.lng })),
      ...(places.data ?? []).map(h => ({ type: 'place',    id: h.id, name: h.name_fr ?? h.name_en ?? h.name_ar })),
      ...(stadiums.data ?? []).map(h => ({ type: 'stadium', id: h.id, name: h.name_fr ?? h.name_en ?? h.name_ar })),
      ...(news.data ?? []).map(h => ({ type: 'news',       id: h.id, name: h.title_fr ?? h.title_en ?? h.title_ar })),
    ];

    return res.status(200).json({ output_text: output, sources });
  } catch (e) {
    const code = e.code === 429 ? 429 : 500;
    console.error('rag-chat error:', e);
    return res.status(code).json({ error: code === 429 ? 'rate_limited' : 'chat_failed' });
  }
}
