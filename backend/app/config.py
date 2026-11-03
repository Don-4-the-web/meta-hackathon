from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    meta_api_key: str = ""
    meta_api_base_url: str = "https://api.meta.ai/v1"
    meta_model: str = "muse-wonder-1.0"
    meta_image_model: str = "muse-image-1.0"
    max_trainings_per_user: int = 5
    training_provider: str = "modal"
    modal_app_name: str = "meta-lora-studio"
    modal_starter_function: str = "start_training"
    modal_generate_function: str = "generate_character"
    modal_volume_name: str = "lora-studio"
    modal_base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    modal_submit_timeout_seconds: float = 180.0
    modal_max_train_steps: int = 400
    gcp_project_id: str = ""
    gcp_region: str = "us-central1"
    gcp_bucket: str = ""
    google_application_credentials: str = ""
    database_url: str = "sqlite:///./chat.db"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_live(self) -> bool:
        return bool(self.meta_api_key.strip())


settings = Settings()
