import { NextResponse } from 'next/server';

export async function POST(req) {
  console.log('=== API-anrop startat: POST /api/chat ===');

  try {
    const body = await req.json();
    console.log('Parsad body:', JSON.stringify(body, null, 2));
    const { message } = body;
    const testReply = `Test-svar från backend: Mottaget "${message || 'inget'}". Kommunikation funkar!`;

    console.log('=== API-anrop lyckades ===');
    return NextResponse.json({ reply: testReply });
  } catch (error) {
    console.error('=== FEL I API ===');
    console.error('Error message:', error.message);
    console.error('Error stack:', error.stack);
    return NextResponse.json({ error: 'Internt serverfel' }, { status: 500 });
  }
}
