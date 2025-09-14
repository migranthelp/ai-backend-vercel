import { GoogleGenerativeAI } from '@google/generative-ai';
import { createClient } from '@supabase/supabase-js';

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const embedModel = genAI.getGenerativeModel({ model: 'text-embedding-004' });

// Primary chat model + fallback
const CHAT_MODEL = process.env.CHAT_MODEL || 'gemini-1.5-flash'; // use flash by default for quota
const PRIMARY = genAI.getGenerativeModel({ model: CHAT_MODEL });
const FALLBACK = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

// Domain guard settings
const STRICT_DOMAIN = process.env.STRICT_DOMAIN === '1';
const MIN_SIM = Number.isFinite(Number(process.env.MIN_SIM)) ? Number(process.env.MIN_SIM) : 0.22;

const sb = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE);

function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
}

const systemStyle = (lang) => `
You are the in-app assistant for "Migrant-e-s Help" in Morocco.
Scope:
- App data: services (cities/categories), news, CAN 2025 stadiums & places.
- Allowed external tools (if allowExternal=true): city weather (Open-Meteo), simple directions, brief web search.
If a question is outside both, refuse briefly and suggest 2–3 in-scope queries.
Answer in ${lang}, concise, bullet points when helpful.
`.trim();

function domainRefusal(lang) {
  switch (lang) {
    case 'fr':
      return "Je peux aider avec les services de l’app, les actualités et CAN 2025. Activez les infos externes pour météo/itinéraires/recherche, ou demandez : « services de santé à Rabat », « stades CAN 2025 », etc.";
    case 'ar':
      return "أساعد في بيانات التطبيق والأخبار ومعلومات كأس إفريقيا 2025. فعّل المصادر الخارجية للمناخ/الاتجاهات/البحث أو جرّب: «خدمات صحية في الرباط»، «ملاعب كان 2025»…";
    default:
      return "I help with app services, news, and CAN 2025. Enable external info for weather/directions/search, or try: “health services in Rabat”, “CAN 2025 stadiums”…";
  }
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
  const result = await chat.sendMessage('Answer the last user message clearly. If you used external info, mention sources briefly.');
  return result.response?.text() ?? '...';
}

/* ---------- Intent Router (cheap rules) ---------- */
function routeIntent(text) {
  const q = (text || '').toLowerCase();

  // weather keywords
  const weatherHits = ['weather', 'météo', 'température', 'forecast', 'pluie', 'rain', 'meteo', ' الطقس', 'طقس'];
  if (weatherHits.some(k => q.includes(k))) return 'weather';

  // directions/transport keywords
  const transportHits = ['bus', 'train', 'tram', 'direction', 'itinéraire', 'itineraire', 'route', 'how to get', 'transport', 'oncf', 'ctm', 'station', 'gare', 'محطة', 'طريق'];
  if (transportHits.some(k => q.includes(k))) return 'directions';

  // generic web search intent
  const webHits = ['news now', 'latest', 'opening hours today', 'happening now', 'prix', 'price', 'reviews', 'review', 'what is', 'who is'];
  if (webHits.some(k => q.includes(k))) return 'web';

  // default to app data
  return 'app_data';
}

/* ---------- External Tools ---------- */
// 1) Weather via Open-Meteo geocoding + forecast (no key)
async function fetchWeather(cityName, lang = 'en') {
  // geocode
  const g = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(cityName)}&count=1&language=${lang}`);
  const gj = await g.json();
  const loc = gj?.results?.[0];
  if (!loc) return { text: null, sources: [] };

  const { latitude, longitude, name, country } = loc;
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,weather_code,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto`;
  const w = await fetch(url);
  const wx = await w.json();

  const cur = wx.current;
  const daily = wx.daily;
  const line1 = `Weather for ${name}, ${country}: ${cur?.temperature_2m}°C, wind ${cur?.wind_speed_10m} km/h.`;
  const line2 = daily ? `Today: min ${daily.temperature_2m_min?.[0]}°C / max ${daily.temperature_2m_max?.[0]}°C, precipitation ${daily.precipitation_sum?.[0]} mm.` : '';
  return {
    text: [line1, line2].filter(Boolean).join('\n'),
    sources: [{ type: 'web', name: 'Open-Meteo', url }]
  };
}

// 2) Directions (stub): you can wire OpenRouteService or Google Directions.
// Here we just return a polite placeholder until you add a key.
async function fetchDirections(queryText, lang = 'en') {
  return {
    text: lang === 'fr'
      ? "Pour les itinéraires en temps réel (bus/train/tram), j’ai besoin d’un service de directions. Ajoutez OPENROUTESERVICE_API_KEY ou GOOGLE_MAPS_API_KEY côté serveur et je donnerai des trajets précis."
      : lang === 'ar'
      ? "للحصول على مسارات دقيقة (حافلة/قطار/ترام)، أحتاج إلى مفتاح لخدمة الاتجاهات. أضِف OPENROUTESERVICE_API_KEY أو GOOGLE_MAPS_API_KEY على الخادم."
      : "For real-time routes (bus/train/tram), please add OPENROUTESERVICE_API_KEY or GOOGLE_MAPS_API_KEY on the server and I’ll provide exact directions.",
    sources: []
  };
}

// 3) Web Search (stub): wire SerpAPI or Google CSE if you want.
async function fetchWebSearch(queryText, lang = 'en') {
  return {
    text: lang === 'fr'
      ? "Recherche web non configurée. Ajoutez SERPAPI_API_KEY ou un Google CSE et j’afficherai 2–3 liens fiables."
      : lang === 'ar'
      ? "البحث على الويب غير مُفعّل. أضِف SERPAPI_API_KEY أو Google CSE وسأعرض 2–3 روابط موثوقة."
      : "Web search not configured. Add SERPAPI_API_KEY or a Google CSE and I’ll return 2–3 reliable links.",
    sources: []
  };
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
    const { chat_id, language = 'en', messages = [], filters = {}, allowExternal = false } = req.body || {};
    const lastUser = [...messages].reverse().find(m => m.role === 'user')?.content || '';
    if (!notEmpty(lastUser)) return res.status(400).json({ error: 'missing_user_message' });

    // Save user turn
    if (chat_id) await sb.from('messages').insert({ chat_id, role: 'user', content: lastUser });

    // Route intent
    const intent = routeIntent(lastUser);

    /* ---------- External branches ---------- */
    if (intent !== 'app_data') {
      if (!allowExternal) {
        return res.status(200).json({ output_text: domainRefusal(language), sources: [] });
      }

      if (intent === 'weather') {
        // Try to extract a city quickly (very naive: last token after "in"/"à"/"في")
        const m = lastUser.match(/\b(?:in|à|في)\s+([A-Za-zÀ-ÿ\u0600-\u06FF\s\-']{2,})$/i);
        const city = m ? m[1].trim() : lastUser;
        const w = await fetchWeather(city, language);
        if (w.text) {
          if (chat_id) await sb.from('messages').insert({ chat_id, role: 'assistant', content: w.text });
          return res.status(200).json({ output_text: w.text, sources: w.sources });
        }
        // fallthrough to app_data if weather failed to find a place
      }

      if (intent === 'directions') {
        const d = await fetchDirections(lastUser, language);
        if (chat_id) await sb.from('messages').insert({ chat_id, role: 'assistant', content: d.text });
        return res.status(200).json({ output_text: d.text, sources: d.sources });
      }

      if (intent === 'web') {
        const s = await fetchWebSearch(lastUser, language);
        if (chat_id) await sb.from('messages').insert({ chat_id, role: 'assistant', content: s.text });
        return res.status(200).json({ output_text: s.text, sources: s.sources });
      }
    }

    /* ---------- App-data RAG (your existing flow) ---------- */
    // 1) Embed
    const er = await embedModel.embedContent({ content: { parts: [{ text: lastUser }] } });
    const qvec = er.embedding.values;

    // 2) RPCs
    const [services, news, stadiums, places] = await Promise.all([
      sb.rpc('match_services',      { query_embedding: qvec, match_count: 6, city_id: filters.cityId ?? null, category_id: filters.categoryId ?? null }),
      sb.rpc('match_news',          { query_embedding: qvec, match_count: 3, category_id: filters.categoryId ?? null }),
      sb.rpc('match_can_stadiums',  { query_embedding: qvec, match_count: 3, city_id: filters.cityId ?? null }),
      sb.rpc('match_places_can',    { query_embedding: qvec, match_count: 4, city_id: filters.cityId ?? null }),
    ]);

    // Similarity gate
    const sims = [
      ...(services.data ?? []).map(x => Number(x.similarity) || 0),
      ...(news.data ?? []).map(x => Number(x.similarity) || 0),
      ...(stadiums.data ?? []).map(x => Number(x.similarity) || 0),
      ...(places.data ?? []).map(x => Number(x.similarity) || 0),
    ];
    const bestSim = sims.length ? Math.max(...sims) : 0;
    const totalMatches = sims.filter(s => s > 0).length;

    if (STRICT_DOMAIN && (bestSim < MIN_SIM || totalMatches === 0)) {
      const txt = domainRefusal(language);
      if (chat_id) await sb.from('messages').insert({ chat_id, role: 'assistant', content: txt });
      return res.status(200).json({ output_text: txt, sources: [] });
    }

    // Build concise context
    const sec = (title, lines) => lines.length ? `\n${title}:\n${lines.join('\n')}` : '';
    const s1 = (services.data ?? []).map(h => `- ${trim(h.name_fr ?? h.name_en ?? h.name_ar ?? 'Service', 80)} @ ${trim(h.address ?? '', 80)} ${h.phone ? `(phone: ${trim(h.phone, 30)})` : ''}`);
    const s2 = (places.data ?? []).map(h => `- ${trim(h.name_fr ?? h.name_en ?? h.name_ar ?? '', 80)} (tourism)`);
    const s3 = (stadiums.data ?? []).map(h => `- ${trim(h.name_fr ?? h.name_en ?? h.name_ar ?? '', 80)} stadium${h.capacity ? `, capacity: ${trim(h.capacity, 20)}` : ''}`);
    const s4 = (news.data ?? []).map(h => `- ${trim(h.title_fr ?? h.title_en ?? h.title_ar ?? '', 100)}`);

    const context =
      sec('Services', s1) +
      sec('Places to visit', s2) +
      sec('CAN 2025 Stadiums', s3) +
      sec('News', s4);

    const history = [
      { role: 'user', parts: [{ text: systemStyle(language) }] },
      { role: 'user', parts: [{ text: 'Context:\n' + (context || 'None') }] },
      ...messages.slice(-2).map(m => ({ role: m.role, parts: [{ text: m.content }]})),
    ];

    // Ask model with fallback
    let output;
    try {
      output = await askModel(PRIMARY, history);
    } catch (e) {
      if (e?.status === 429) {
        const waitMs = getRetryDelayMs(e);
        await new Promise(r => setTimeout(r, waitMs));
        output = await askModel(FALLBACK, history);
      } else {
        throw e;
      }
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
    console.error(e);
    if (e?.status === 429) return res.status(429).json({ error: 'rate_limited', message: 'Please wait a few seconds and try again.' });
    return res.status(500).json({ error: 'chat_failed' });
  }
}
