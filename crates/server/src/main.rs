//! TensorWarp server binary — OpenAI-compatible inference endpoint.
//!
//! Configuration via environment variables:
//!   HOST  — bind address (default: 0.0.0.0)
//!   PORT  — listen port  (default: 8000)
//!   MODEL — model name   (default: tensorwarp-test)

#[tokio::main]
async fn main() {
    env_logger::init();

    let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".into());
    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "8000".into())
        .parse()
        .unwrap_or(8000);
    let model = std::env::var("MODEL").unwrap_or_else(|_| "tensorwarp-test".into());
    let model_path = std::env::var("MODEL_PATH").ok();

    if let Err(e) = tensorwarp_server::serve(&host, port, &model, model_path.as_deref()).await {
        eprintln!("Server error: {e}");
        std::process::exit(1);
    }
}
