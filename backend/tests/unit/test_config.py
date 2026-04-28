import json
import uuid

from news_service.core.config import Settings


def test_settings_parses_json_pricing_table_into_prices() -> None:
    model_name = f"openai/{uuid.uuid4().hex}"
    settings = Settings(
        openai_api_key=f"sk-{uuid.uuid4().hex}",
        yandex_search_api_key=uuid.uuid4().hex,
        litellm_model=model_name,
        litellm_embedding_model=model_name,
        litellm_judge_model=model_name,
        llm_model_pricing_usd_per_1m=json.dumps(
            {
                model_name: {
                    "input": 0.11,
                    "output": 0.29,
                },
            }
        ),
    )

    assert settings.llm_model_pricing_usd_per_1m[model_name]["output"] == 0.29, (
        "pricing JSON was not parsed into numeric prices"
    )
