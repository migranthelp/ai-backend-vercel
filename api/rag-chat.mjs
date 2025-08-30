import { GoogleGenerativeAI } from '@google/generative-ai';
import { createClient } from '@supabase/supabase-js';

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const chatModel = genAI.getGenerativeModel({ model: 'gemini-1.5-pro' });
const embedModel = genAI.getGenerativeModel({ model: 'text-embedding-004' });

const sb = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE);

function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
}

const systemStyle = (lang) => `
You are a helpful assistant for the Migrant-e-s Help app.
Answer in ${lang}, be concise, and prefer bullet points.
Use the provided context (services/news/stadiums/places). If unsure, ask a brief follow-up.
`.trim();

async function saveMessage(chatId, role, content) {
  if (!chatId) return;
  await sb.from('messages').insert({ chat_id: chatId, role, content });
}

export default async function handler(req, res) {
  cors(res);
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    const { chat_id, language = 'en', messages = [], filters = {} } = req.body || {};
    const lastUser = [...messages].reverse().find(m => m.role === 'user')?.content || '';
    if (chat_id && lastUser) await saveMessage(chat_id, 'user', lastUser);

    // 1) Embed query
    const er = await embedModel.embedContent({ content: { parts: [{ text: lastUser }] } });
    const qvec = er.embedding.values;

    // 2) RAG across your RPCs (ensure you created them in Supabase)
    const [services, news, stadiums, places] = await Promise.all([
      sb.rpc('match_services', { query_embedding: qvec, match_count: 8, city_id: filters.cityId ?? null, category_id: filters.categoryId ?? null }),
      sb.rpc('match_news',     { query_embedding: qvec, match_count: 4, category_id: filters.categoryId ?? null }),
      sb.rpc('match_can_stadiums', { query_embedding: qvec, match_count: 4, city_id: filters.cityId ?? null }),
      sb.rpc('match_places_can',   { query_embedding: qvec, match_count: 6, city_id: filters.cityId ?? null }),
    ]);

    // 3) Build short context
    const sec = (title, lines) => lines.length ? `\n${title}:\n${lines.join('\n')}` : '';
    const s1 = (services.data ?? []).map(h => `- ${h.name_fr ?? h.name_en ?? h.name_ar ?? 'Service'} @ ${h.address ?? ''} (phone: ${h.phone ?? ''})`);
    const s2 = (places.data ?? []).map(h => `- ${h.name_fr ?? h.name_en ?? h.name_ar ?? ''} (tourism)`);
    const s3 = (stadiums.data ?? []).map(h => `- ${h.name_fr ?? h.name_en ?? h.name_ar ?? ''} stadium, capacity: ${h.capacity ?? ''}`);
    const s4 = (news.data ?? []).map(h => `- ${h.title_fr ?? h.title_en ?? h.title_ar ?? ''}`);

    const context =
      sec('Services', s1) +
      sec('Places to visit', s2) +
      sec('CAN 2025 Stadiums', s3) +
      sec('News', s4);

    // 4) Model call (history = system + context + current turn)
    const history = [
      { role: 'user', parts: [{ text: systemStyle(language) }] },
      { role: 'user', parts: [{ text: 'Context:' + (context || ' None') }] },
      ...messages.map(m => ({ role: m.role, parts: [{ text: m.content }]})),
    ];

    const chat = await chatModel.startChat({ history });
    const result = await chat.sendMessage('Answer the last user message clearly and cite names.');
    const output = result.response?.text() ?? '...';

    if (chat_id) await saveMessage(chat_id, 'assistant', output);

    const sources = [
      ...(services.data ?? []).map(h => ({ type: 'service',  id: h.id, name: h.name_fr ?? h.name_en ?? h.name_ar, lat: h.lat, lng: h.lng })),
      ...(places.data ?? []).map(h => ({ type: 'place',    id: h.id, name: h.name_fr ?? h.name_en ?? h.name_ar })),
      ...(stadiums.data ?? []).map(h => ({ type: 'stadium', id: h.id, name: h.name_fr ?? h.name_en ?? h.name_ar })),
      ...(news.data ?? []).map(h => ({ type: 'news',     id: h.id, name: h.title_fr ?? h.title_en ?? h.title_ar })),
    ];

    return res.status(200).json({ output_text: output, sources });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: 'chat_failed' });
  }
}
