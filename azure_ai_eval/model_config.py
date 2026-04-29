from azure.ai.evaluation import AzureOpenAIModelConfiguration
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureOpenAIChatClientServiceEnv(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    azure_openai_endpoint: str
    azure_openai_deployed_model_name: str
    azure_openai_api_version: str
    azure_openai_api_key: str | None = None


env = AzureOpenAIChatClientServiceEnv()  # type: ignore
model_config = AzureOpenAIModelConfiguration(
    azure_endpoint=env.azure_openai_endpoint,
    api_version=env.azure_openai_api_version,
    azure_deployment=env.azure_openai_deployed_model_name,
)
