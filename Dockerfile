# Multi-stage Dockerfile para Credit Risk Scoring Engine en Rust
FROM rust:1.80-slim as builder

WORKDIR /usr/src/credit_risk_engine

# Dependencias del sistema necesarias para compilar Polars y C-bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Compilación en release con optimizaciones LTO
RUN cargo build --release

# Runtime image ligera
FROM debian:bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/credit_risk_engine/target/release/credit_risk_engine /app/credit_risk_engine
COPY dashboard /app/dashboard
COPY data /app/data

EXPOSE 3000

ENV RUST_LOG=info

CMD ["/app/credit_risk_engine", "serve", "--port", "3000"]
