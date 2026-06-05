from flask import Flask, request, jsonify, render_template
from groq import Groq
import os, json

app = Flask(__name__)

SYSTEM_PROMPT = """You are a content moderation AI. Analyze the given text and classify it accurately.

Classification rules:
- "safe": normal conversation, greetings, questions, friendly messages, educational content
- "spam": promotional content, unsolicited ads, repetitive messages, scam-like content
- "toxic": hate speech, threats, content targeting people with extreme negativity
- "offensive": crude language, mild insults, inappropriate but not hateful content

Be accurate — most everyday text should be classified as "safe".
A simple greeting like "hello" or "how are you" is ALWAYS safe.

Respond ONLY with a JSON object in this exact format, nothing else:
{
  "classification": "safe|spam|toxic|offensive",
  "confidence": {"safe": 0.0, "spam": 0.0, "toxic": 0.0, "offensive": 0.0},
  "reason": "brief explanation"
}
All confidence values must add up to 1.0."""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/moderate', methods=['POST'])
def moderate():
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return jsonify({'error': 'API key not configured'}), 500
        client = Groq(api_key=api_key)
        data = request.get_json()
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this text: {text}"}
            ]
        )
        result = response.choices[0].message.content
        clean = result.strip().replace('```json', '').replace('```', '').strip()
        parsed = json.loads(clean)
        return jsonify(parsed)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
