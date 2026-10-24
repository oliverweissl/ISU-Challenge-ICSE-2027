FROM isu-challenge-base

RUN pip install diffusers==0.35.1 sentencepiece protobuf safetensors

# FLUX.1-schnell (Apache-2.0), transformer stored as fp8 (~12 GB), text encoders/VAE bf16 (~9 GB): ~21 GB in total.
COPY build_weights.py /opt/build_weights.py
RUN python /opt/build_weights.py /opt/flux-schnell && chmod -R a+rX /opt/flux-schnell
