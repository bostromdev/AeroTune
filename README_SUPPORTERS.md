# AeroTune Local Analyzer Access

Thank you for supporting AeroTune.

This private supporter repo is for approved AeroTune supporters who want to run the local analyzer version on their own machine.

Local analysis is best for larger Betaflight Blackbox logs, testing without web upload limits, and using AeroTune when the public web version is slow or limited by hosting.

Public web version:

    https://aerotune.onrender.com

Private local repo:

    https://github.com/AeroTuneLabs/AeroTune-Local

---

## What This Access Includes

- Local AeroTune analyzer repo access
- Setup instructions for running AeroTune locally
- Ability to analyze larger logs on your own machine
- Access to local analyzer updates while this repo remains active
- Community support through the AeroTune supporter Discord

---

## What This Access Does Not Include

This repo access does not include:

- Private one-on-one tuning support
- Unlimited setup support
- Guaranteed log review
- Commercial resale rights
- Rehosting rights
- Permission to copy AeroTune into another paid product
- Permission to redistribute the source code

Private tuning help and full drone support are separate support options.

---

## Basic Local Setup

Clone the repo:

    git clone https://github.com/AeroTuneLabs/AeroTune-Local.git
    cd AeroTune-Local

Create a Python virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Upgrade pip and install requirements:

    python -m pip install --upgrade pip
    pip install -r requirements.txt

Run AeroTune locally:

    uvicorn app.main:app --reload

Then open this in your browser:

    http://127.0.0.1:8000

---

## Recommended Tuning Workflow

For best results:

1. Start with a clean stock or known-good baseline flight.
2. Make sure the quad is mechanically healthy.
3. Record a Blackbox log.
4. Run the log through AeroTune.
5. Make one controlled tune change at a time.
6. Fly again using the same general route.
7. Check motor temperature after every flight.
8. Compare before and after logs.
9. Keep, reduce, revert, or retest based on the data and flight feel.

Do not blindly stack PID or filtering changes.

AeroTune is built to help pilots make controlled tuning decisions from real flight data instead of guessing.

---

## Important Safety Notes

AeroTune does not magically create a perfect tune from one log.

Every drone build is different. Frame stiffness, prop size, motor size, ESC firmware, gyro noise, filtering, battery condition, and pilot style all matter.

Always:

- Remove props when testing on the bench.
- Check motor direction before flying.
- Check motor temperature after tuning changes.
- Avoid large PID jumps.
- Avoid blindly increasing D-term.
- Revert changes if motors get hot or the quad sounds noisy.
- Fly in a safe area away from people, roads, and property.

You are responsible for your own aircraft and flight safety.

---

## Access Terms

This repo is for personal/local use by approved AeroTune supporters only.

Access does not grant permission to:

- Resell AeroTune
- Rehost AeroTune as another public or paid service
- Redistribute the source code
- Copy AeroTune logic into another paid product
- Remove attribution
- Claim the project as your own

Commercial use, redistribution, resale, rehosting, or derivative paid products require written permission from the AeroTune maintainer.

A full license file will be added as AeroTune licensing is finalized.

---

## Support

For general community discussion, use the AeroTune supporter Discord.

For private log review or one-on-one tuning help, use the separate AeroTune support options.
