import { GoogleSpreadsheet } from 'google-spreadsheet';
import { Pinecone } from '@pinecone-database/pinecone';
import { HfInference } from '@huggingface/inference';
import crypto from 'crypto';

const doc = new GoogleSpreadsheet('1DskBGn-cvbEn30NKBpyeueOvowB8-YagnTACz9LIChk');

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pinecone.index(process.env.PINECONE_INDEX_NAME);

const hf = new HfInference(process.env.HF_TOKEN);

async function syncKB() {
  try {
    // Autentisera Google Sheet
    await doc.useServiceAccountAuth({
      client_email: process.env.GOOGLE_SERVICE_ACCOUNT_EMAIL,
      private_key: process.env.GOOGLE_PRIVATE_KEY.replace(/\\n/g, '\n'),
    });

    await doc.loadInfo();

    // ÄNDRA HÄR om data inte ligger på första fliken
    const sheet = doc.sheetsByIndex[0]; // eller doc.sheetsByTitle['Fliknamn']

    const rows = await sheet.getRows();

    // Radera alla gamla vectors först (håller indexet rent)
    try {
      await index.delete({ deleteAll: true });
      console.log('Raderade alla gamla vectors');
    } catch (deleteError) {
      console.warn('Kunde inte radera alla (tomt index eller SDK-begränsning?):', deleteError.message);
    }

    const vectorsToUpsert = [];

    for (const row of rows) {
      // ANPASSA HÄR: Exakta kolumnnamn från er Sheet (case-sensitive!)
      const question = (row.get('Fråga') || row.get('Question') || '').trim();
      const answer = (row.get('Svar') || row.get('Answer') || '').trim();
      const category = (row.get('Kategori') || row.get('Category') || 'allmänt').trim();

      if (!question || !answer) continue; // Hoppa över tomma rader

      const passageText = `${question}\n\n${answer}`;

      // Generera embedding (passage-prefix för e5-modellen)
      const embeddingResponse = await hf.featureExtraction({
        model: 'intfloat/multilingual-e5-large',
        inputs: `passage: ${passageText}`,
      });

      const embedding = Array.from(embeddingResponse);

      // Stabilt ID baserat på innehåll
      const id = crypto.createHash('sha256').update(passageText).digest('hex');

      vectorsToUpsert.push({
        id,
        values: embedding,
        metadata: {
          question,
          answer,
          category,
          text: passageText, // Detta används i /api/chat för context – nu med både fråga och svar
        },
      });
    }

    // Upsert i batcher (Pinecone gillar max ~100–500 per call)
    if (vectorsToUpsert.length > 0) {
      for (let i = 0; i < vectorsToUpsert.length; i += 100) {
        const batch = vectorsToUpsert.slice(i, i + 100);
        await index.upsert(batch);
      }
      console.log(`Synk klar! Upsertade ${vectorsToUpsert.length} FAQ-entryer.`);
    } else {
      console.log('Inga rader att synka.');
    }

    return { success: true, count: vectorsToUpsert.length };
  } catch (error) {
    console.error('Synk fel:', error);
    return { success: false, error: error.message };
  }
}

export default async function handler(req, res) {
  // Tillåt både GET (för Vercel cron) och POST (för manuell trigger)
  if (req.method === 'GET' || req.method === 'POST') {
    const result = await syncKB();
    res.status(result.success ? 200 : 500).json(result);
  } else {
    res.status(405).json({ error: 'Method not allowed – använd GET eller POST' });
  }
}
