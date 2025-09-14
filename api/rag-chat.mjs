// api/rag-chat.mjs
import { GoogleGenerativeAI } from '@google/generative-ai';
import { createClient } from '@supabase/supabase-js';

/* ---------- Models & Clients ---------- */
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const embedModel = genAI.getGenerativeModel({ model: 'text-embedding-004' });

const CHAT_MODEL = process.env.CHAT_MODEL || 'gemini-1.5-flash'; // flash default = friendlier quotas
const PRIMARY = genAI.getGenerativeModel({ model: CHAT_MODEL });
const FALLBACK = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

const sb = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE);

const STRICT_DOMAIN = process.env.STRICT_DOMAIN === '1';
const MIN_SIM = Number.isFinite(Number(process.env.MIN_SIM)) ? Number(process.env.MIN_SIM) : 0.22;

const ORS_KEY = process.env.OPENROUTESERVICE_API_KEY;   // optional
const SERPAPI_KEY = process.env.SERPAPI_API_KEY;        // optional
const FRONTEND_APP_KEY = process.env.FRONTEND_APP_KEY;  // optional simple gate

/* ---------- Helpers ---------- */
function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-APP-KEY');
}

const systemStyle = (lang) => `
You are the in-app assistant for "Migrant-e-s Help" in Morocco.
Primary scope: app data (services + cities/categories, news, CAN 2025 stadiums & places).
If allowExternal=true you may also answer: weather for a city, simple directions, brief web lookups.
Stay concise, answer in ${lang}, prefer bullet points. If insufficient context, ask a brief follow-up.
`.trim();

function domainRefusal(lang) {
  switch (lang) {
    case 'fr':
      return "Je réponds en priorité avec les données de l’app (services, actualités, CAN 2025). Activez les infos externes pour la météo/itinéraires/recherche.";
    case 'ar':
      return "أجيب أولًا من بيانات التطبيق (الخدمات، الأخبار، كأس إفريقيا 2025). فعّل المصادر الخارجية للمناخ/الاتجاهات/البحث.";
    default:
      return "I answer primarily from the app’s data (services, news, CAN 2025). Enable external info for weather/directions/search.";
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
  const result = await chat.sendMessage('Answer the last user message clearly. If you used external info, mention it briefly.');
  return result.response?.text() ?? '...';
}

/* ---------- Intent (cheap rules) ---------- */
function routeIntent(text) {
  const q = (text || '').toLowerCase();

  const weather = ['weather','météo','meteo','forecast','pluie','rain','température','طقس','الطقس'];
  if (weather.some(k => q.includes(k))) return 'weather';

  const dirs = ['bus','train','tram','direction','itinéraire','itineraire','route','transport','oncf','ctm','station','gare','محطة','طريق','كيف أصل','from ',' to '];
  if (dirs.some(k => q.includes(k))) return 'directions';

  const web = ['latest','now','opening hours','prix','price','reviews','review','what is','who is','خبر الآن','أحدث'];
  if (web.some(k => q.includes(k))) return 'web';

  return 'app_data';
}



  // 2) WEATHER (word-level, so "train" != "rain")
  const weatherTokens = [
    'weather','météo','meteo','forecast','pluie','température','طقس','الطقس','raining','rain'
  ];
  if (hasAny(toks, weatherTokens)) return 'weather';

  // 3) WEB
  const webPhrases = ['opening hours','latest','now','prix','price','reviews','review','what is','who is','خبر الآن','أحدث'];
  if (webPhrases.some(p => containsPhrase(q, p))) return 'web';

  // 4) default
  return 'app_data';
}


/* ---------- External: Weather (Open-Meteo) ---------- */
async function fetchWeather(cityName, lang = 'en') {
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
  return { text: [line1, line2].filter(Boolean).join('\n'), sources: [{ type: 'web', name: 'Open-Meteo', url }] };
}

/* ---------- External: Directions (OpenRouteService) ---------- */
function parseFromTo(text) {
  const t = (text || '').trim();
  let m = t.match(/\bfrom\s+(.+?)\s+to\s+(.+?)$/i);               // EN
  if (m) return { from: m[1].trim(), to: m[2].trim() };
  m = t.match(/\bde\s+(.+?)\s+(?:à|a)\s+(.+?)$/i);                 // FR
  if (m) return { from: m[1].trim(), to: m[2].trim() };
  m = t.match(/من\s+(.+?)\s+إلى\s+(.+)$/i);                        // AR
  if (m) return { from: m[1].trim(), to: m[2].trim() };
  return null;
}

async function geocodeORS(query) {
  const url = `https://api.openrouteservice.org/geocode/search?api_key=${ORS_KEY}&text=${encodeURIComponent(query)}&size=1`;
  const r = await fetch(url);
  const j = await r.json();
  const f = j?.features?.[0];
  if (!f) return null;
  const [lng, lat] = f.geometry.coordinates;
  return { lat, lng, name: f.properties.label };
}

async function directionsORS(origin, dest, profile = 'driving-car', lang = 'en') {
  const url = `https://api.openrouteservice.org/v2/directions/${profile}?api_key=${ORS_KEY}&language=${lang}`;
  const body = { coordinates: [[origin.lng, origin.lat],[dest.lng, dest.lat]], instructions: true, units: 'km' };
  const r = await fetch(url, { method: 'POST', headers: {'content-type':'application/json'}, body: JSON.stringify(body) });
  const j = await r.json();
  if (!j?.features?.[0]) return null;

  const s = j.features[0].properties?.segments?.[0];
  const dist = (s.distance/1000).toFixed(1);
  const durMin = Math.round(s.duration/60);
  const steps = (s.steps || []).slice(0, 6).map((st, i) => `${i+1}. ${st.instruction}`);
  return {
    text: `Route: ${origin.name} → ${dest.name}\nDistance: ${dist} km, Duration: ~${durMin} min\n\nSteps:\n` + steps.join('\n'),
    sources: [{ type:'web', name:'OpenRouteService', url:'https://openrouteservice.org/' }]
  };
}

async function fetchDirectionsSmart(queryText, lang='en') {
  if (!ORS_KEY) {
    return {
      text: lang === 'fr'
        ? "Ajoutez OPENROUTESERVICE_API_KEY côté serveur pour obtenir des itinéraires réels (voiture/marche)."
        : lang === 'ar'
        ? "أضِف OPENROUTESERVICE_API_KEY على الخادم للحصول على مسارات فعلية (سيارة/سير)."
        : "Add OPENROUTESERVICE_API_KEY on the server to get real routes (car/walking).",
      sources: []
    };
  }
  const ft = parseFromTo(queryText);
  if (!ft) {
    return {
      text: lang === 'fr'
        ? "Indiquez: « de [origine] à [destination] » (ex: de Rabat à Casablanca)."
        : lang === 'ar'
        ? "اكتب: « من [المكان] إلى [الوجهة] » (مثال: من الرباط إلى الدار البيضاء)."
        : "Use: “from [origin] to [destination]” (e.g., from Rabat to Casablanca).",
      sources: []
    };
  }
  const [o, d] = await Promise.all([geocodeORS(ft.from), geocodeORS(ft.to)]);
  if (!o || !d) return { text: "Couldn't geocode origin/destination. Try clearer place names.", sources: [] };
  return await directionsORS(o, d, 'driving-car', lang) || { text: "No route found.", sources: [] };
}

/* ---------- External: Web (SerpAPI) ---------- */
async function fetchWebSerp(queryText, lang='en') {
  if (!SERPAPI_KEY) {
    return {
      text: lang === 'fr'
        ? "Recherche web non configurée. Ajoutez SERPAPI_API_KEY."
        : lang === 'ar'
        ? "البحث على الويب غير مُفعّل. أضِف SERPAPI_API_KEY."
        : "Web search not configured. Add SERPAPI_API_KEY.",
      sources: []
    };
  }
  const params = new URLSearchParams({
    engine: 'google', q: queryText, hl: lang || 'en', gl: 'ma', num: '5', api_key: SERPAPI_KEY
  });
  const r = await fetch(`https://serpapi.com/search.json?${params.toString()}`);
  const j = await r.json();
  const items = (j.organic_results || []).slice(0,3).map(o => `• ${o.title}\n  ${o.link}\n  ${o.snippet || ''}`);
  return {
    text: items.length ? `Top results:\n\n${items.join('\n\n')}` : 'No relevant results.',
    sources: [{ type:'web', name:'Google (SerpAPI)', url:'https://serpapi.com' }]
  };
}

/* ---------- Handler ---------- */
export default async function handler(req, res) {
  cors(res);
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'method_not_allowed' });

  // Optional simple gate
  if (FRONTEND_APP_KEY) {
    const got = req.headers['x-app-key'];
    if (!got || got !== FRONTEND_APP_KEY) return res.status(401).json({ error: 'unauthorized' });
  }

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

    // Intent
    const intent = routeIntent(lastUser);

    /* ---------- External branches ---------- */
    if (intent !== 'app_data') {
      if (!allowExternal) {
        const txt = domainRefusal(language);
        if (chat_id) await sb.from('messages').insert({ chat_id, role: 'assistant', content: txt });
        return res.status(200).json({ output_text: txt, sources: [] });
      }

      if (intent === 'weather') {
        // naive city grab: try token after "in/à/في", else whole query
        const m = lastUser.match(/\b(?:in|à|في)\s+([A-Za-zÀ-ÿ\u0600-\u06FF\s\-']{2,})$/i);
        const city = m ? m[1].trim() : lastUser;
        const w = await fetchWeather(city, language);
        if (w.text) {
          if (chat_id) await sb.from('messages').insert({ chat_id, role: 'assistant', content: w.text });
          return res.status(200).json({ output_text: w.text, sources: w.sources });
        }
        // fall through to app_data if no city found
      }

      if (intent === 'directions') {
        const d = await fetchDirectionsSmart(lastUser, language);
        if (chat_id) await sb.from('messages').insert({ chat_id, role: 'assistant', content: d.text });
        return res.status(200).json({ output_text: d.text, sources: d.sources });
      }

      if (intent === 'web') {
        const s = await fetchWebSerp(lastUser, language);
        if (chat_id) await sb.from('messages').insert({ chat_id, role: 'assistant', content: s.text });
        return res.status(200).json({ output_text: s.text, sources: s.sources });
      }
    }

    /* ---------- App-data RAG ---------- */
    // 1) Embed
    const er = await embedModel.embedContent({ content: { parts: [{ text: lastUser }] } });
    const qvec = er.embedding.values;

    // 2) RPCs (tune counts)
    const [services, news, stadiums, places] = await Promise.all([
      sb.rpc('match_services',      { query_embedding: qvec, match_count: 6, city_id: filters.cityId ?? null, category_id: filters.categoryId ?? null }),
      sb.rpc('match_news',          { query_embedding: qvec, match_count: 3, category_id: filters.categoryId ?? null }),
      sb.rpc('match_can_stadiums',  { query_embedding: qvec, match_count: 3, city_id: filters.cityId ?? null }),
      sb.rpc('match_places_can',    { query_embedding: qvec, match_count: 4, city_id: filters.cityId ?? null }),
    ]);
    if (services.error)  console.error('services rpc error', services.error);
    if (news.error)      console.error('news rpc error', news.error);
    if (stadiums.error)  console.error('stadiums rpc error', stadiums.error);
    if (places.error)    console.error('places rpc error', places.error);

    // 3) Similarity gate
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

    // 4) Context
    const sec = (title, lines) => lines.length ? `\n${title}:\n${lines.join('\n')}` : '';
    const s1 = (services.data ?? []).map(h => `- ${trim(h.name_fr ?? h.name_en ?? h.name_ar ?? 'Service', 80)} @ ${trim(h.address ?? '', 80)} ${h.phone ? `(phone: ${trim(h.phone, 30)})` : ''}`);
    const s2 = (places.data ?? []).map(h => `- ${trim(h.name_fr ?? h.name_en ?? h.name_ar ?? '', 80)} (tourism)`);
    const s3 = (stadiums.data ?? []).map(h => `- ${trim(h.name_fr ?? h.name_en ?? h.name_ar ?? '', 80)} stadium${h.capacity ? `, capacity: ${trim(h.capacity, 20)}` : ''}`);
    const s4 = (news.data ?? []).map(h => `- ${trim(h.title_fr ?? h.title_en ?? h.title_ar ?? '', 100)}`);
    const context = sec('Services', s1) + sec('Places to visit', s2) + sec('CAN 2025 Stadiums', s3) + sec('News', s4);

    const history = [
      { role: 'user', parts: [{ text: systemStyle(language) }] },
      { role: 'user', parts: [{ text: 'Context:\n' + (context || 'None') }] },
      ...messages.slice(-2).map(m => ({ role: m.role, parts: [{ text: m.content }]})),
    ];

    // 5) Model w/ fallback
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
