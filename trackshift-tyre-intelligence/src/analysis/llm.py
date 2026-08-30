"""
LLM Integration for automated post-run analysis summaries.
Uses OpenRouter API to generate insights based on model outputs.
"""
import os
import logging
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Keys pulled securely from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
ENDPOINT = os.getenv("OPENAI_ENDPOINT", "https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENAI_MODEL", "qwen/qwen-2.5-7b-instruct")

def stream_ai_summary(
    driver: str,
    stint_label: str,
    compound: str,
    raw_laps: int,
    valid_laps: int,
    naive_rate: float,
    multi_rate: float,
    final_rate: float,
    total_deg: float,
    model_used: str,
    sampling_time: float,
):
    """Generate a streaming AI summary of the analysis run."""
    try:
        client = OpenAI(api_key=API_KEY, base_url=ENDPOINT)
        
        system_prompt = (
            "You are an expert F1 Race Engineer and Principal Data Scientist. "
            "Your job is to impress a panel of hackathon judges by summarizing the results of our 'Tyre Degradation Intelligence' system. "
            "Speak clearly, authoritatively, and professionally. Use markdown for emphasis and structure. "
            "Keep it concise (3-4 short paragraphs maximum)."
        )
        
        # Determine the fallback narrative
        fallback_narrative = ""
        if "Kalman" in model_used or "MLE" in model_used:
            fallback_narrative = (
                "Crucially, note that the heavy Bayesian MCMC model hit a computation/memory limit, "
                "but our fault-tolerant 'Inference Orchestrator' instantly caught the crash and fell back to a deterministic "
                "Kalman Filter (MLE). This is a massive engineering win for robustness—the app didn't crash, it adapted in sub-seconds."
            )
        else:
            fallback_narrative = (
                "Note that the full Bayesian State-Space (MCMC) model successfully ran to completion, "
                "sampling the posterior space to give us rigorous mathematical uncertainty bounds."
            )

        user_prompt = f"""
        Here is the data from the latest analysis run:
        - Driver: {driver}
        - Stint: {stint_label} (Compound: {compound})
        
        Model Outputs:
        1. Naive Model (Raw LapTime vs Age): {naive_rate:+.4f} seconds/lap. (If negative, it looks like the car is getting faster due to fuel burn!).
        2. Multivariate Model (w/ Fuel, Temp, Track evolution): {multi_rate:+.4f} seconds/lap.
        3. Final Engine Model Used: {model_used}
        4. Final True Degradation Rate Extracted: {final_rate:+.4f} seconds/lap.
        5. Total Latent Degradation over stint: {total_deg:+.3f} seconds.
        6. Execution Time: {sampling_time:.2f} seconds.
        
        {fallback_narrative}
        
        Write a highly impressive, executive-level summary. Include:
        1. What just happened in the run.
        2. The "Paradox" (why the naive model was wrong/confounded by fuel).
        3. The Architecture/Fallback (what model executed and how it proved fault-tolerant or rigorous).
        4. The Final Finding (what the true tyre wear actually is).
        """

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=True,
            temperature=0.3,
            max_tokens=500
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        yield f"⚠️ **AI Communication Link Failed:** Unable to reach the LLM endpoint. (Error: {str(e)})"
