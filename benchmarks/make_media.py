"""Generate synthetic "meeting" videos for benchmarking.

Speech is synthesised with the local Windows SAPI voices (en-US / pt-BR) and laid out
with realistic pauses: short gaps between sentences, occasional long silences. The
resulting speech density is printed and stored next to the file, because it drives ASR
cost far more than duration does.

    python benchmarks/make_media.py --minutes 10 --language en benchmarks/media/meeting-10m-en.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from support import SR, silence, tts, write_media

SENTENCES = {
    "en": [
        "Let's start with the status of the deployment pipeline.",
        "The release failed on Friday because the authentication secret expired.",
        "We switched the App Service to managed identity to avoid that in the future.",
        "The database migration took about forty minutes, which is longer than planned.",
        "Can someone check the monitoring dashboard for errors after the rollout?",
        "The latency of the search endpoint increased by thirty percent last week.",
        "I think the cache configuration is the main suspect here.",
        "We should add an alert when the error rate goes above two percent.",
        "Marketing asked whether the new onboarding flow will ship this month.",
        "The customer in Lisbon reported problems with the invoice export.",
        "Let's move the retrospective to Thursday afternoon.",
        "The budget for cloud costs is already eighty percent used.",
    ],
    "pt": [
        "Vamos começar pelo status do pipeline de implantação.",
        "A versão falhou na sexta-feira porque o segredo de autenticação expirou.",
        "Migramos o App Service para identidade gerenciada para evitar isso no futuro.",
        "A migração do banco de dados levou cerca de quarenta minutos.",
        "Alguém pode verificar o painel de monitoramento depois da implantação?",
        "A latência do endpoint de busca aumentou trinta por cento na semana passada.",
        "Acho que a configuração do cache é a principal suspeita.",
        "Precisamos de um alerta quando a taxa de erros passar de dois por cento.",
        "O cliente de Lisboa relatou problemas na exportação de notas fiscais.",
        "Vamos mover a retrospectiva para quinta-feira à tarde.",
    ],
}


def build(minutes: float, language: str, seed: int = 0) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    voices = [tts(s, language) for s in SENTENCES[language]]
    target = int(minutes * 60 * SR)
    parts: list[np.ndarray] = [silence(1.0)]
    total = len(parts[0])
    speech = 0
    while total < target:
        clip = voices[int(rng.integers(len(voices)))]
        parts.append(clip)
        speech += len(clip)
        roll = rng.random()
        pause = (
            rng.uniform(0.2, 1.2)
            if roll < 0.8
            else rng.uniform(2, 6)
            if roll < 0.97
            else rng.uniform(20, 60)
        )
        parts.append(silence(pause))
        total += len(clip) + int(pause * SR)
    audio = np.concatenate(parts)[:target]
    return audio, speech / len(audio)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--language", default="en", choices=sorted(SENTENCES))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    audio, density = build(args.minutes, args.language, args.seed)
    write_media(args.output, audio, fps=2, size=(64, 48))
    meta = {"minutes": args.minutes, "language": args.language, "speech_density": round(density, 3)}
    args.output.with_suffix(".json").write_text(json.dumps(meta))
    print(f"{args.output}: {meta}")


if __name__ == "__main__":
    main()
