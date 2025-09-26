export const config = { runtime: 'nodejs' }; // ensures Node runtime, not Edge
import { GoogleGenerativeAI } from '@google/generative-ai';
import { createClient } from '@supabase/supabase-js';
import crypto from 'crypto';
//
/* ---------- Env & Clients ---------- */
const FRONTEND_APP_KEY = process.env.FRONTEND_APP_KEY;            // REQUIRED
const MAX_REQ_PER_IP_DAY = Number(process.env.MAX_REQ_PER_IP_DAY || 200);
const MAX_MSG_LEN = Number(process.env.MAX_MSG_LEN || 1200);
const MAX_CONTEXT_LEN = 3000; // clamp context length

if (!process.env.GOOGLE_API_KEY) throw new Error('missing GOOGLE_API_KEY');
if (!process.env.SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE) {
  throw new Error('missing Supabase env');
}
if (!FRONTEND_APP_KEY) throw new Error('missing FRONTEND_APP_KEY');

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const embedModel = genAI.getGenerativeModel({ model: 'text-embedding-004' });
const CHAT_MODEL = process.env.CHAT_MODEL || 'gemini-1.5-flash';
const PRIMARY = genAI.getGenerativeModel({ model: CHAT_MODEL });

const sb = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE);

/* ---------- CORS (tighten origin in prod) ---------- */
function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', process.env.ALLOWED_ORIGIN || '*'); 
  // 👆 set ALLOWED_ORIGIN in env, e.g. "https://migrantapp.example"
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-APP-KEY');
}

/* ---------- Tiny rate limit ---------- */
async function enforceRateLimit(req, lang) {
  const ip = (req.headers['x-forwarded-for'] || req.socket?.remoteAddress || '').split(',')[0].trim();
  const day = new Date().toISOString().slice(0,10);
  await sb.from('request_log').insert({ day, ip }).select();
  const { count } = await sb.from('request_log')
    .select('*', { head: true, count: 'exact' })
    .eq('day', day).eq('ip', ip);
  if ((count || 0) > MAX_REQ_PER_IP_DAY) {
    const messages = {
      en: "⚠️ Too many requests today. Please try again tomorrow.",
      fr: "⚠️ Trop de requêtes aujourd'hui. Réessayez demain.",
      ar: "⚠️ عدد كبير من الطلبات اليوم. حاول مرة أخرى غداً."
    };
    const e = new Error(messages[lang] || messages.en);
    e.code = 429;
    throw e;
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

/* ---------- System prompt in multiple languages ---------- */
const systemStyle = (lang) => {
  switch (lang) {
    case 'fr': return `
Vous êtes l’assistant intégré de « Migrant-e-s Help » (Maroc).
Répondez UNIQUEMENT à partir des données de l’application (services, villes/catégories, actualités, stades CAN 2025).
Style : concis, en français, sous forme de puces, inclure noms/téléphones/adresses si pertinents.
Si l’info manque, dites-le clairement sans inventer.
`.trim();
    case 'ar': return `
أنت المساعد داخل تطبيق "Migrant-e-s Help" في المغرب.
أجب فقط من بيانات التطبيق (الخدمات، المدن/الفئات، الأخبار، ملاعب كأس إفريقيا 2025).
الأسلوب: مختصر، بالعربية، على شكل نقاط، مع ذكر الأسماء/الهاتف/العناوين إن وُجدت.
إذا لم تتوفر المعلومات، صرّح بعدم توفرها دون اختلاق.
`.trim();
    default: return `
You are the in-app assistant for "Migrant-e-s Help" (Morocco).
Answer ONLY from app data (services, cities/categories, news, CAN 2025).
Style: concise, in English, bullet points, include names/phones/addresses when relevant.
If info is missing, say you don't have it. Do not invent.
`.trim();
  }
};

const languagePreference = (lang, map) => {
  const order = map[lang] || map.en;
  return (record) => {
    for (const key of order) {
      const value = record?.[key];
      if (value) return value;
    }
    return '';
  };
};

const pickServiceName = (lang) => languagePreference(lang, {
  en: ['name_en', 'name_fr', 'name_ar'],
  fr: ['name_fr', 'name_en', 'name_ar'],
  ar: ['name_ar', 'name_fr', 'name_en'],
});

const pickPlaceName = pickServiceName;
const pickStadiumName = pickServiceName;
const pickNewsTitle = (lang) => languagePreference(lang, {
  en: ['title_en', 'title_fr', 'title_ar'],
  fr: ['title_fr', 'title_en', 'title_ar'],
  ar: ['title_ar', 'title_fr', 'title_en'],
});

/* ---------- Handler ---------- */
export default async function handler(req, res) {
  cors(res);
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'method_not_allowed' });

  // Require app key
  const gotKey = req.headers['x-app-key'];
  if (gotKey !== FRONTEND_APP_KEY) return res.status(401).json({ error: 'unauthorized' });

  try {
    const { chat_id, language = 'en', messages = [], filters = {} } = req.body || {};
    await enforceRateLimit(req, language);

    let lastUser = [...messages].reverse().find(m => m.role === 'user')?.content || '';
    lastUser = String(lastUser).trim().slice(0, MAX_MSG_LEN);
    if (!lastUser) return res.status(400).json({ error: 'missing_user_message' });

    if (chat_id) await sb.from('messages').insert({ chat_id, role: 'user', content: lastUser });

    // 1) Embed (cached)
    const qvec = await embedOnce(lastUser);

    // 2) Query Supabase RPCs
    const [services, news, stadiums, places] = await Promise.all([
      sb.rpc('match_services',     { query_embedding: qvec, match_count: 6, city_id: filters.cityId ?? null, category_id: filters.categoryId ?? null }),
      sb.rpc('match_news',         { query_embedding: qvec, match_count: 3, category_id: filters.categoryId ?? null }),
      sb.rpc('match_can_stadiums', { query_embedding: qvec, match_count: 3, city_id: filters.cityId ?? null }),
      sb.rpc('match_places_can',   { query_embedding: qvec, match_count: 4, city_id: filters.cityId ?? null }),
    ]);

    // ✅ Log errors
    if (services.error) console.error('services rpc error', services.error);
    if (news.error) console.error('news rpc error', news.error);
    if (stadiums.error) console.error('stadiums rpc error', stadiums.error);
    if (places.error) console.error('places rpc error', places.error);

    const serviceName = pickServiceName(language);
    const placeName = pickPlaceName(language);
    const stadiumName = pickStadiumName(language);
    const newsTitle = pickNewsTitle(language);

    // 3) Build context (clamped)
    const trim = (s, n=300) => (s || '').toString().slice(0, n);
    const sec = (title, lines) => lines.length ? `\n${title}:\n${lines.join('\n')}` : '';

    const s1 = (services.data ?? []).map(h => `- ${trim(serviceName(h) || 'Service',80)} @ ${trim(h.address ?? '',80)} ${h.phone ? `(☎ ${trim(h.phone,30)})` : ''}`);
    const s2 = (places.data ?? []).map(h => `- ${trim(placeName(h),80)} (tourism)`);
    const s3 = (stadiums.data ?? []).map(h => `- ${trim(stadiumName(h),80)} stadium${h.capacity ? `, capacity: ${trim(h.capacity,20)}` : ''}`);
    const s4 = (news.data ?? []).map(h => `- ${trim(newsTitle(h),100)}`);

    let context = sec('Services', s1) + sec('Places', s2) + sec('Stadiums', s3) + sec('News', s4);
    context = context.slice(0, MAX_CONTEXT_LEN);

    const history = [
      { role: 'user', parts: [{ text: systemStyle(language) }] },
      { role: 'user', parts: [{ text: 'Context:\n' + (context || 'None') }] },
      ...messages.slice(-2).map(m => ({ role: m.role, parts: [{ text: String(m.content || '').slice(0, MAX_MSG_LEN) }] })),
    ];

    // 4) Ask model (with one retry)
    let output;
    try {
      const chat = await PRIMARY.startChat({ history });
      const result = await chat.sendMessage('Answer the last user message from the provided context ONLY.');
      output = result.response?.text() ?? '...';
    } catch (err) {
      if (err?.status === 429) {
        // wait 2s and retry once
        await new Promise(r => setTimeout(r, 2000));
        try {
          const chat = await PRIMARY.startChat({ history });
          const result = await chat.sendMessage('Answer the last user message from the provided context ONLY.');
          output = result.response?.text() ?? '...';
        } catch {
          output = language === 'fr'
            ? "⚠️ Limite de requêtes atteinte. Réessayez plus tard."
            : language === 'ar'
            ? "⚠️ تم تجاوز الحد. حاول لاحقاً."
            : "⚠️ Rate limit reached. Try again later.";
        }
      } else {
        throw err;
      }
    }

    if (chat_id) await sb.from('messages').insert({ chat_id, role: 'assistant', content: output });

    const sources = [
      ...(services.data ?? []).map(h => ({ type: 'service', id: h.id, name: serviceName(h) })),
      ...(places.data ?? []).map(h => ({ type: 'place', id: h.id, name: placeName(h) })),
      ...(stadiums.data ?? []).map(h => ({ type: 'stadium', id: h.id, name: stadiumName(h) })),
      ...(news.data ?? []).map(h => ({ type: 'news', id: h.id, name: newsTitle(h) })),
    ];

    return res.status(200).json({ output_text: output, sources });
  } catch (e) {
    const code = e.code === 429 ? 429 : 500;
    return res.status(code).json({ error: code === 429 ? 'rate_limited' : 'chat_failed', message: e.message });
  }
}
try {
  const er = await embedModel.embedContent({ content: { parts: [{ text: lastUser }] } });
  const qvec = er.embedding.values;
} catch (e) {
  console.error('EMBED ERROR', e?.status, e?.message, e); // shows 429/403/etc.
  return res.status(500).json({ error: 'embed_failed', message: String(e?.message || e) });
}

const [services, news, stadiums, places] = await Promise.all([
  sb.rpc('match_services', {...}).then(r => (r.error && console.error('RPC services', r.error), r)),
  sb.rpc('match_news', {...}).then(r => (r.error && console.error('RPC news', r.error), r)),
  sb.rpc('match_can_stadiums', {...}).then(r => (r.error && console.error('RPC stadiums', r.error), r)),
  sb.rpc('match_places_can', {...}).then(r => (r.error && console.error('RPC places', r.error), r)),
]);
