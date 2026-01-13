import { Pinecone } from '@pinecone-database/pinecone';
import { HfInference } from '@huggingface/inference';
import { google } from 'googleapis';
import crypto from 'crypto';

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pinecone.index(process.env.PINECONE_INDEX_NAME);

const hf = new HfInference(process.env.HF_TOKEN);

async function syncKB() {
  try {
    // Officiella Google API – funkar perfekt på Vercel
    const auth = new google.auth.GoogleAuth({
      credentials: {
        client_email: process.env.GOOGLE_SERVICE_ACCOUNT_EMAIL,
        private_key: process.env.GOOGLE_PRIVATE_KEY.replace(/\\n/g, '\n'),
      },
      scopes: ['https://www.googleapis.com/auth/spreadsheets.readonly'],
    });

    const sheets = google.sheets({ version: 'v4', auth });

    // Hämta alla rader från Sheet:en
    const response = await sheets.spreadsheets.values.get({
      spreadsheetId: '1DskBGn-cvbEn30NKBpyeueOvowB8-YagnTACz9LIChk',
      range: 'A:Z', // Anpassa om ni har specifik flik, t.ex. 'Fliknamn!A:Z'
    });

    const rows = response.data.values || [];

    if (rows.length === 0) {
      console.log('Inga rader i Sheet:en.');
      return { success: true, count: 0 };
    }

    // Antag header i rad 1, data från rad 2
    const header = rows[0];
    const dataRows = rows.slice(1);

    try {
      await index.delete({ deleteAll: true });
      console.log('Raderade alla gamla vectors');
    } catch (e) {
      console.warn('Kunde inte radera alla:', e.message);
    }

    const vectorsToUpsert = [];

    for (const row of dataRows) {
      // Skapa objekt från header
      const rowObj = header.reduce((obj, key, i) => {
        obj[key] = row[i] || '';
        return obj;
      }, {});

      const question = (rowObj['Fråga'] || rowObj['Question'] || '').trim();
      const answer = (rowObj['Svar'] || rowObj['Answer'] || '').trim();
      const category = (rowObj['Kategori'] || rowObj['Category'] || 'allmänt').trim();

      if (!question || !answer) continue;

      const passageText = `${question}\n\n${answer}`;

      const embeddingResponse = await hf.featureExtraction({
        model: 'intfloat/multilingual-e5-large',
        inputs: `passage: ${passageText}`,
      });

      const embedding = Array.from(embeddingResponse);

      const id = crypto.createHash('sha256').update(passageText).digest('hex');

      vectorsToUpsert.push({
        id,
        values: embedding,
        metadata: {
          question,
          answer,
          category,
          text: passageText,
        },
      });
    }

    if (vectorsToUpsert.length > 0) {
      for (let i = 0; i < vectorsToUpsert.length; i += 100) {
        const batch = vectorsToUpsert.slice(i, i + 100);
        await index.upsert(batch);
      }
      console.log(`Synk klar! Upsertade ${vectorsToUpsert.length} entryer.`);
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
  if (req.method === 'GET' || req.method === 'POST') {
    const result = await syncKB();
    res.status(result.success ? 200 : 500).json(result);
  } else {
    res.status(405).json({ error: 'Method not allowed' });
  }
}
