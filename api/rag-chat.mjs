import { GoogleGenerativeAI } from '@google/generative-ai';
import { createClient } from '@supabase/supabase-js';

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const embedModel = genAI.getGenerativeModel({ model: 'text-embedding-004' });

// Primary chat model + fallback
const CHAT_MODEL = process.env.CHAT_MODEL || 'gemini-1.5-pro';
const PRIMARY = genAI.getGenerativeModel({ model: CHAT_MODEL });
const FALLBACK = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

// Domain guard settings
const STRICT_DOMAIN = process.env.STRICT_DOMAIN === '1';
const MIN_SIM = Number.isFinite(Number(process.env.MIN_SIM))
  ? Number(process.env.MIN_SIM)
  : 0.22; // default threshold

const sb = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE);

function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
}

// Clear, strict domain policy injected into the prompt.
const systemStyle = (lang) => `
You are the in-app assistant for "Migrant-e-s Help".
Your scope is strictly:
- services (with cities/categories) from the app’s database,
- news articles in the app,
- CAN 2025 stadiums and places.

RULES:
1) Answer ONLY within that scope, using the supplied context.
2) If the user asks anything outside this scope, reply with a brief refusal and suggest 2–3 in-scope queries.
3) Be concise, answer in ${lang}, and prefer bullet points.
`.trim();

// Short, localized refusal used by the hard retrieval gate
function domainGuardMessage(lang = 'en') {
  switch (lang) {
    case 'fr':
      return "Je suis conçu pour aider uniquement sur les services de l’application, les actualités et les informations CAN 2025. Essayez, par exemple : « Services de santé à Rabat », « Actualités catégorie X », ou « Stades CAN 2025 à Casablanca ».";
    case 'ar':
      return "أنا مخصّص للإجابة فقط عن خدمات التطبيق، الأخبار، ومعلومات كأس إفريقيا 2025. جرّب مثلًا: «خدمات صحية في الرباط»، «أخبار الفئة X»، أو «ملاعب كأس إفريقيا 2025 في الدار البيضاء».";
    default:
      return "I only help with the app’s services, news, and CAN 2025 info. Try: “health services in Rabat”, “news in category X”, or “CAN 2025 stadiums in Casablanca”.";
  }
}

async function saveMessage(chatId, role, content) {
  if (!chatId) return;
  await sb.from('messages').insert({ chat_id: chatId, role, content });
}

const trim = (s, n = 300) => (s || '').toString().slice(0, n);
const notEmpty = (x) => x != null && String(x).trim().length > 0;

function getRetryDelayMs(err) {
  const details = err?.errorDetails || [];
  const retry = details.find(d => d['@type']?.includes('RetryInfo'));
  if (retry?.retryDelay) {
    const m = /(\d+)s/.exec(retry.retryDelay);
    if (m) return Number(m[1]) * 1000;
  }
  return 4000;
}

async function askModel(model, history) {
  const chat = await model.startChat({ history });
  const result = await chat.sendMessage('Answer the last user message clearly and cite names.');
  return result.response?.text() ?? '...';
}

export default async function handler(req, res) {
  cors(res);
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'method_not_allowed' });

  if (!process.env.GOOGLE_API_KEY) return res.status(500).json({ error: 'missing_GOOGLE_API_KEY' });
  if (!process.env.SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE) {
    return res.status(500).json({ error: 'missing_supabase_env' });
  }

  try {
    const { chat_id, language = 'en', messages = [], filters = {} } = req.body || {};
    const lastUser = [...messages].reverse().find(m => m.role === 'user')?.content || '';
    if (!notEmpty(lastUser)) return res.status(400).json({ error: 'missing_user_message' });

    if (chat_id) await saveMessage(chat_id, 'user', lastUser);

    // 1) Embed query
    const er = await embedModel.embedContent({ content: { parts: [{ text: lastUser }] } });
    const qvec = er.embedding.values;

    // 2) RAG across RPCs (keep token use reasonable)
    const [services, news, stadiums, places] = await Promise.all([
      sb.rpc('match_services',      { query_embedding: qvec, match_count: 6, city_id: filters.cityId ?? null, category_id: filters.categoryId ?? null }),
      sb.rpc('match_news',          { query_embedding: qvec, match_count: 3, category_id: filters.categoryId ?? null }),
      sb.rpc('match_can_stadiums',  { query_embedding: qvec, match_count: 3, city_id: filters.cityId ?? null }),
      sb.rpc('match_places_can',    { query_embedding: qvec, match_count: 4, city_id: filters.cityId ?? null }),
    ]);

    // Log RPC errors (don’t fail the whole request)
    if (services.error)  console.error('services rpc error', services.error);
    if (news.error)      console.error('news rpc error', news.error);
    if (stadiums.error)  console.error('stadiums rpc error', stadiums.error);
    if (places.error)    console.error('places rpc error', places.error);

    // 2b) HARD DOMAIN GATE: if no relevant matches (by similarity), refuse before LLM
    const sims = [
      ...(services.data ?? []).map(x => Number(x.similarity) || 0),
      ...(news.data ?? []).map(x => Number(x.similarity) || 0),
      ...(stadiums.data ?? []).map(x => Number(x.similarity) || 0),
      ...(places.data ?? []).map(x => Number(x.similarity) || 0),
    ];
    const bestSim = sims.length ? Math.max(...sims) : 0;
    const totalMatches = sims.filter(s => s > 0).length;

    if (STRICT_DOMAIN && (bestSim < MIN_SIM || totalMatches === 0)) {
      // Don’t call the model at all
      return res.status(200).json({
        output_text: domainGuardMessage(language),
        sources: []
      });
    }

    // 3) Build short context (trimmed lines)
    const sec = (title, lines) => lines.length ? `\n${title}:\n${lines.join('\n')}` : '';
    const s1 = (services.data ?? []).map(h =>
      `- ${trim(h.name_fr ?? h.name_en ?? h.name_ar ?? 'Service', 80)} @ ${trim(h.address ?? '', 80)} ${h.phone ? `(phone: ${trim(h.phone, 30)})` : ''}`
    );
    const s2 = (places.data ?? []).map(h =>
      `- ${trim(h.name_fr ?? h.name_en ?? h.name_ar ?? '', 80)} (tourism)`
    );
    const s3 = (stadiums.data ?? []).map(h =>
      `- ${trim(h.name_fr ?? h.name_en ?? h.name_ar ?? '', 80)} stadium${h.capacity ? `, capacity: ${trim(h.capacity, 20)}` : ''}`
    );
    const s4 = (news.data ?? []).map(h =>
      `- ${trim(h.title_fr ?? h.title_en ?? h.title_ar ?? '', 100)}`
    );

    const context =
      sec('Services', s1) +
      sec('Places to visit', s2) +
      sec('CAN 2025 Stadiums', s3) +
      sec('News', s4);

    // 4) Lightweight history (system + context + last up to 2 turns)
    const lightHistory = [
      { role: 'user', parts: [{ text: systemStyle(language) }] },
      { role: 'user', parts: [{ text: 'Context:\n' + (context || 'None') }] },
      ...messages.slice(-2).map(m => ({ role: m.role, parts: [{ text: m.content }]})),
    ];

    // 5) Call model with quota fallback
    let output;
    try {
      output = await askModel(PRIMARY, lightHistory);
    } catch (e) {
      if (e?.status === 429) {
        const waitMs = getRetryDelayMs(e);
        await new Promise(r => setTimeout(r, waitMs));
        try {
          output = await askModel(FALLBACK, lightHistory);
        } catch (e2) {
          if (e2?.status === 429) {
            return res.status(429).json({ error: 'rate_limited', message: 'Please wait a few seconds and try again.' });
          }
          throw e2;
        }
      } else {
        throw e;
      }
    }

    if (chat_id) await saveMessage(chat_id, 'assistant', output);

    const sources = [
      ...(services.data ?? []).map(h => ({ type: 'service',  id: h.id, name: h.name_fr ?? h.name_en ?? h.name_ar, lat: h.lat, lng: h.lng })),
      ...(places.data ?? []).map(h => ({ type: 'place',    id: h.id, name: h.name_fr ?? h.name_en ?? h.name_ar })),
      ...(stadiums.data ?? []).map(h => ({ type: 'stadium', id: h.id, name: h.name_fr ?? h.name_en ?? h.name_ar })),
      ...(news.data ?? []).map(h => ({ type: 'news',       id: h.id, name: h.title_fr ?? h.title_en ?? h.title_ar })),
    ];

    return res.status(200).json({ output_text: output, sources });
  } catch (e) {
    console.error(e);
    if (e?.status === 429) {
      return res.status(429).json({ error: 'rate_limited', message: 'Please wait a few seconds and try again.' });
    }
    return res.status(500).json({ error: 'chat_failed' });
  }
}
