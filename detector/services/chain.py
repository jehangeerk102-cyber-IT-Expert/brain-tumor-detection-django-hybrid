import json
from django.conf import settings

SYSTEM_PROMPT = """You summarize an image-classification result for a software demo. Never diagnose, prescribe, or claim certainty. Explain that the result is an AI prediction and that a qualified radiologist/doctor must review the original scan. Keep the response concise and use the supplied class and confidence only."""

def explain_prediction(prediction: dict) -> dict:
    fallback = (
        f"AI prediction: {prediction['result']} with {prediction['confidence'] * 100:.2f}% confidence. "
        "This is not a medical diagnosis. Please have the original scan reviewed by a qualified medical professional."
    )
    if not settings.LANGCHAIN_ANALYSIS_ENABLED or not settings.OPENAI_API_KEY:
        return {"summary": fallback, "provider": "fallback"}
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "Prediction JSON: {prediction}")])
        chain = prompt | ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0)
        response = chain.invoke({"prediction": json.dumps(prediction)})
        return {"summary": response.content, "provider": "langchain-openai"}
    except Exception:
        return {"summary": fallback, "provider": "fallback"}
