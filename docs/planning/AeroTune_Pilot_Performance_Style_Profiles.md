# AeroTune Pilot Performance Style Profiles

## Purpose

This document defines the first draft of a separate AeroTune feature area:

**Pilot Performance Analysis**

This should remain separate from the core PID/tune recommendation engine.

The main AeroTune system should answer:

> What should change in the tune?

The Pilot Performance Analysis system should answer:

> How is the pilot flying, and how well does that match the intended style?

These two systems can share evidence from the same Blackbox log, but they should not be treated as the same thing.

---

# Core Concept

AeroTune should eventually let the user select one of five flying styles before analysis.

The selected style changes how the log is interpreted.

The same flight behavior may be good for one style and bad for another.

Example:

- Smooth throttle is ideal for cinematic and long range.
- Hard throttle changes are normal for aggressive freestyle.
- Fast stick inputs are expected in racing.
- Flow and controlled recovery matter most in smooth freestyle.

---

# Five Pilot Style Profiles

## 1. Cinematic

### Description

Cinematic flying prioritizes smooth camera movement, controlled throttle, low oscillation, stable horizon behavior, and efficient flight.

This style is not about sharp snap response. It is about clean, steady, visually pleasing motion.

### Priorities

- Smooth throttle control
- Minimal visible wobble
- Stable attitude during slow turns
- Controlled descents
- Low motor heat
- Efficient battery usage
- Low propwash disruption
- Predictable stick response

### Analyzer Should Reward

- Low throttle jerk
- Smooth roll/pitch/yaw rates
- Low bounceback
- Stable GPS speed
- Consistent altitude behavior if altitude data exists
- Low gyro noise
- Good battery efficiency

### Analyzer Should Warn About

- Jerky throttle inputs
- Overcorrection after turns
- Oscillation during cruising
- Excessive stick snap
- High motor noise or heat risk
- Tune that is too aggressive for smooth video

### Tune Bias

- Moderate or lower feedforward
- Conservative D increases
- Smooth throttle curve
- Stable filtering
- Avoid aggressive response unless instability is present

---

## 2. Smooth Freestyle

### Description

Smooth Freestyle sits between Cinematic and Aggressive Freestyle.

It is flow-based freestyle: tricks, rolls, flips, dives, and reversals are allowed, but the goal is controlled style rather than violent snap.

This is likely the best profile for large freestyle/long-range hybrid builds.

### Priorities

- Flowing movement
- Controlled tricks
- Smooth recovery after flips/rolls
- Good propwash handling
- Predictable throttle response
- Balanced stick sharpness
- Moderate locked-in feel without excessive motor heat

### Analyzer Should Reward

- Clean recovery after flips and rolls
- Low bounceback after maneuvers
- Controlled throttle unloads
- Smooth direction changes
- Stable attitude after stick release
- Balanced gyro/setpoint tracking
- Good flow without constant overcorrection

### Analyzer Should Warn About

- Floaty recovery after maneuvers
- Wobble during throttle unload
- Excessively soft feedforward
- Too much D heat risk
- Excessive filtering delay
- Overcorrection after tricks
- Propwash during descents

### Tune Bias

- Moderate feedforward increase when stick feel is soft
- Small D/D Max increases when bounceback or underdamping exists
- Avoid large D increases when motors are heat-sensitive
- Prefer feedforward over extra D when the issue is stick authority, not damping
- Mild throttle expo may help large/heavy builds

### Important AeroTune Rule

If the pilot wants a more locked-in feel but motor heat risk is already elevated, AeroTune should not blindly push D higher.

Instead:

- Use modest feedforward increases for stick authority.
- Use smaller D/D Max increases for true bounceback/settling problems.
- Cap D recommendations when motor heat/noise risk is high.
- Avoid adding excessive filtering just to hide too much D.

---

## 3. Aggressive Freestyle

### Description

Aggressive Freestyle prioritizes snap response, fast reversals, hard stops, strong propwash recovery, and direct stick feel.

This style is more demanding on motors, ESCs, batteries, and filtering.

### Priorities

- Immediate response
- Sharp tracking
- Minimal delay
- Strong propwash recovery
- Fast flip/roll recovery
- Low bounceback
- High authority during reversals

### Analyzer Should Reward

- Fast setpoint tracking
- Low delay between stick input and gyro response
- Minimal bounceback
- Stable recovery after high-rate maneuvers
- Strong throttle transition control
- Good high-throttle stability

### Analyzer Should Warn About

- Excessive motor heat
- High D-term noise
- Battery sag during aggressive maneuvers
- Overly high feedforward causing twitchiness
- Excessive propwash
- Mechanical noise from loose frame/props
- Too much filtering delay

### Tune Bias

- Higher feedforward allowed
- More aggressive D/D Max adjustments allowed if motors stay cool
- Less smoothing than cinematic/smooth freestyle
- Stronger propwash correction
- Tighter bounceback detection

---

## 4. Racing

### Description

Racing prioritizes precision, latency, line-holding, fast corrections, and predictable high-speed behavior.

It is not mainly about smooth video or freestyle style.

### Priorities

- Fast response
- Precise setpoint tracking
- Low latency
- Stable high-speed turns
- Strong throttle authority
- Minimal overshoot
- Consistent line holding

### Analyzer Should Reward

- Low tracking error
- Fast response to stick input
- Clean high-speed cornering
- Low oscillation at speed
- Minimal bounceback
- Consistent throttle-to-speed response

### Analyzer Should Warn About

- Delay from too much filtering
- Soft feedforward
- High-speed oscillation
- Poor line holding
- Battery sag limiting punch-outs
- Excessive throttle smoothness reducing control

### Tune Bias

- Sharper feedforward
- Lower smoothing if signal is clean
- Stronger focus on delay metrics
- More tolerance for aggressive feel
- Still cap recommendations when heat/noise is unsafe

---

## 5. Long Range / Cruising

### Description

Long Range / Cruising prioritizes efficiency, reliability, predictable rescue behavior, low heat, and stable flight over aggressive response.

This profile is especially important when GPS data is available.

### Priorities

- Efficiency
- Low motor heat
- Stable cruising
- Low battery sag
- Predictable throttle behavior
- Reliable GPS Rescue
- Smooth turns
- Strong signal/recovery awareness

### Analyzer Should Reward

- Low current draw at cruise
- Stable GPS speed
- Efficient throttle-to-speed behavior
- Low oscillation during cruise
- Predictable descent control
- Good battery reserve behavior
- Low heat risk

### Analyzer Should Warn About

- High current draw for low speed
- Excessive D-term heat risk
- Overloaded prop/motor setup
- Severe voltage sag
- Unstable throttle unloads
- Poor rescue climb authority
- High vibration/noise

### Tune Bias

- Conservative D/D Max
- Moderate feedforward only if needed
- More filtering allowed if it improves reliability without causing bad delay
- Throttle curve recommendations may matter more than aggressive PID changes
- Strong battery/mAh/voltage warnings

---

# GPS Data Value

GPS data should eventually feed the Pilot Performance Analysis section.

Useful GPS-backed metrics may include:

- Ground speed
- Speed stability
- Distance traveled
- mAh per mile
- Throttle versus speed
- Cruise efficiency
- High-speed oscillation detection
- Propwash during fast descents
- Rescue climb behavior
- Return-to-home behavior
- Wind/load inference

GPS data should not replace gyro/setpoint analysis.

It should add real-world movement context.

---

# Blackbox Logging Recommendations

For normal AeroTune users:

- 1/16 logging rate = recommended default
- 1/8 logging rate = advanced analysis
- 1/4 logging rate = high-detail local analysis
- 1/2 logging rate = developer diagnostic logging
- 1/1 logging rate = deep testing only, not normal use

For the hosted web version:

- Recommend 1/16.
- Allow 1/8 if file size is manageable.
- Warn users that high logging rates may create files too large for hosted analysis.

For local AeroTune:

- 1/4 is a strong high-detail option.
- 1/2 can be used for focused developer testing.

GPS data should not dramatically increase log size because GPS updates much slower than gyro/PID loop data.

Expected GPS size increase is likely modest compared to logging rate changes.

---

# Battery Safety Notes

For the user's current 1600mAh pack behavior:

- Around 1100–1150mAh: start planning return.
- Around 1200–1250mAh: stop aggressive freestyle and head back.
- Around 1300mAh: land immediately.
- Around 1350mAh+: emergency reserve only.
- Around 1380mAh: known unsafe thrust-collapse region on this setup.

AeroTune should eventually support battery safety warnings based on:

- mAh consumed
- voltage sag
- current draw
- GPS distance from home
- flight style
- throttle reserve

---

# Key Tuning Principle To Add Later

For more locked-in feel without extra motor heat:

Feedforward is often the better first move when the issue is stick authority.

D gain fixes bounceback, settling, and damping problems.

Feedforward sharpens stick response and can improve locked-in feel without heating motors as much as pushing D too far.

AeroTune should distinguish:

## Stick Authority Problem

Symptoms:

- Soft center feel
- Delayed response to input
- Feels floaty but does not bounce back much
- Low-to-moderate overshoot

Likely recommendation:

- Increase feedforward slightly
- Reduce excessive feedforward smoothing if needed
- Avoid large D increase

## Damping Problem

Symptoms:

- Bounceback after flips/stops
- Wobble after throttle unload
- Poor settling after maneuvers
- Propwash recovery problems

Likely recommendation:

- Small D increase
- Small D Max increase
- Watch motor heat/noise
- Avoid over-filtering

## Heat-Risk Condition

Symptoms:

- Motors already hot
- High D-term noise
- Battery/current stress
- Large prop/heavy build

Likely recommendation:

- Cap D increases
- Prefer small FF increase if stick feel is soft
- Consider throttle shaping
- Consider filter review only if noise supports it

---

# Recommended UI Structure

AeroTune should eventually show:

## Section 1: Tune Analysis

- PID delta recommendations
- D/D Max guidance
- Feedforward guidance
- Filtering guidance
- Motor heat/noise warnings

## Section 2: Pilot Performance Analysis

- Selected style
- Style match score
- Throttle control score
- Stick smoothness score
- Recovery control score
- Propwash handling score
- Efficiency score if GPS/current data exists
- Safety warnings

## Section 3: Flight Context

- Logging rate
- GPS availability
- Battery size
- Drone size
- Prop size
- Motor size/KV
- Weight if known
- Pilot notes
- Weather/wind notes

---

# Initial Five Style Names

Use these in the first UI version:

1. Cinematic
2. Smooth Freestyle
3. Aggressive Freestyle
4. Racing
5. Long Range / Cruising

Smooth Freestyle should be described as:

> Flow-based freestyle that balances controlled tricks, smooth recovery, moderate locked-in feel, and predictable throttle behavior without becoming overly aggressive or cinematic-soft.

---

# Implementation Reminder

Do not let style selection blindly stack huge PID changes.

Style preference should bias interpretation, not override safety.

Hard caps must still apply.

Examples:

- Do not raise D aggressively if heat risk is high.
- Do not increase feedforward aggressively if the quad is already twitchy.
- Do not recommend cinematic smoothing if the log shows dangerous underdamping.
- Do not recommend racing-style sharpness on a heat-limited heavy 7-inch build without warnings.

AeroTune should stay conservative, evidence-based, and safety-first.
