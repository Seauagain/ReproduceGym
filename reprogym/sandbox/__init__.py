"""Host-side sandbox runtime (steps 4-6).

  launcher  start sandbox on the HOST (reuse ClawGym chroot/docker backend),
            mount input_files/, inject the API key from .env
  runner    issue the task user_query to the in-sandbox reproduction agent,
            let it reach remote GPUs via plain ssh when needed, record trajectory
  retry     resume the conversation after an interruption

reward/ is never mounted here; scoring happens out-of-band in reprogym.verify.
Stubs only.
"""
