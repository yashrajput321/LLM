import os
from dotenv import load_dotenv
import gradio as gr
from google import genai
from google.genai import types

# -----------------------------
# Load API Key securely
# -----------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API")

if not API_KEY:
    raise ValueError("❌ ERROR: GEMINI_API key not found in .env file")

client = genai.Client(api_key=API_KEY)

# -----------------------------
# Persona templates
# -----------------------------
personalities = {
    "Friendly": """You are a friendly, enthusiastic, and highly encouraging Study Assistant.
    Break down concepts simply, use examples, and always ask a follow-up question.""",

    "Academic": """You are a strict academic professor.
    Give structured, formal, detailed explanations.
    Always ask a follow-up question."""
}

# -----------------------------
# LLM Function
# -----------------------------
def study_assistant(question, persona):
    system_prompt = personalities[persona]

    full_prompt = f"{system_prompt}\n\nUser Question: {question}"

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        config=types.GenerateContentConfig(temperature=0.4),
        contents=full_prompt
    )

    return response.text


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Study Assistant") as demo:
    gr.Markdown("## 📘 AI Study Assistant\nChoose your persona and start chatting!")

    persona_selector = gr.Radio(
        choices=list(personalities.keys()),
        value="Friendly",
        label="Choose Persona"
    )

    chatbot = gr.Chatbot(height=450)

    msg = gr.Textbox(
        placeholder="Ask a question...",
        label="Your Question"
    )

    clear = gr.Button("Clear Chat")

    # --- FIXED FUNCTION FOR GRADIO 6.x ---
    def respond(user_input, chat_history, persona):
        answer = study_assistant(user_input, persona)

        # convert to correct format
        chat_history = chat_history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": answer}
        ]

        return "", chat_history

    msg.submit(respond, [msg, chatbot, persona_selector], [msg, chatbot])

    def clear_chat():
        return []

    clear.click(clear_chat, None, chatbot)

demo.launch(debug=True)