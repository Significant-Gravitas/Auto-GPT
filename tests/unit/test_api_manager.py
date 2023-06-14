from unittest.mock import patch

import pytest

from autogpt.llm.api_manager import COSTS, ApiManager

api_manager = ApiManager()


@pytest.fixture(autouse=True)
def reset_api_manager():
    api_manager.reset()
    yield


@pytest.fixture(autouse=True)
def mock_costs():
    with patch.dict(
        COSTS,
        {
            "gpt-3.5-turbo-0613": {"prompt": 0.002, "completion": 0.002},
            "text-embedding-ada-002": {"prompt": 0.0004, "completion": 0},
        },
        clear=True,
    ):
        yield


class TestApiManager:
    def test_getter_methods(self):
        """Test the getter methods for total tokens, cost, and budget."""
        api_manager.update_cost(60, 120, "gpt-3.5-turbo-0613")
        api_manager.set_total_budget(10.0)
        assert api_manager.get_total_prompt_tokens() == 60
        assert api_manager.get_total_completion_tokens() == 120
        assert api_manager.get_total_cost() == (60 * 0.002 + 120 * 0.002) / 1000
        assert api_manager.get_total_budget() == 10.0

    @staticmethod
    def test_set_total_budget():
        """Test if setting the total budget works correctly."""
        total_budget = 10.0
        api_manager.set_total_budget(total_budget)

        assert api_manager.get_total_budget() == total_budget

    @staticmethod
    def test_update_cost():
        """Test if updating the cost works correctly."""
        prompt_tokens = 50
        completion_tokens = 100
        model = "gpt-3.5-turbo-0613"

        api_manager.update_cost(prompt_tokens, completion_tokens, model)

        assert api_manager.get_total_prompt_tokens() == 50
        assert api_manager.get_total_completion_tokens() == 100
        assert api_manager.get_total_cost() == (50 * 0.002 + 100 * 0.002) / 1000

    @staticmethod
    def test_get_models():
        """Test if getting models works correctly."""
        with patch("openai.Model.list") as mock_list_models:
            mock_list_models.return_value = {"data": [{"id": "gpt-3.5-turbo-0613"}]}
            result = api_manager.get_models()

            assert result[0]["id"] == "gpt-3.5-turbo-0613"
            assert api_manager.models[0]["id"] == "gpt-3.5-turbo-0613"
