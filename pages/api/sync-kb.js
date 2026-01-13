import { Pinecone } from '@pinecone-database/pinecone';
import { HfInference } from '@huggingface/inference';
import crypto from 'crypto';

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pinecone.index(process.env.PINECONE_INDEX_NAME);

const hf = new HfInference(process.env.HF_TOKEN);

async function syncKB() {
  try {
    // Dynamisk import för google-spreadsheet (fixar bundling-felet)
    const { GoogleSpreadsheet } = await import('google-spreadsheet');

    const doc = new GoogleSpreadsheet('1DskBGn-cvbEn30NKBpyeueOvowB8-YagnTACz9LIChk');

    await doc.useServiceAccountAuth({
      client_email: process.env.GOOGLE_SERVICE_ACCOUNT_EMAIL,
      private_key: process.env.GOOGLE_PRIVATE_KEY.replace(/\\n/g, '\n'),
    });

    await doc.loadInfo();

    const sheet = doc.sheetsByIndex[0]; // Ändra vid behov

    const rows = await sheet.getRows();

    try {
      await index.delete({ deleteAll: true });
      console.log('Raderade alla gamla vectors');
    } catch (e) {
      console.warn('Kunde inte radera alla:', e.message);
    }

    const vectorsToUpsert = [];

    for (const row of rows) {
      const question = (row.get('Fråga') || row.get('Question') || '').trim();
      const answer = (row.get('Svar') || row.get('Answer') || '').trim();
      const category = (row.get('Kategori') || row.get('Category') || 'allmänt').trim();

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
