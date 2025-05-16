# ✅ RL Trader: Finalized Design for Real-Time Distributed RL System

## 🎯 Objective

Design a production-ready, real-time reinforcement learning (RL) system with fully decoupled, stateless inference and clear modular separation for maintainability, performance, and scalability.

---

## 📦 Modules

### 1. **Inference Module**

**Purpose:** Stateless, fast consumer of pre-trained models that only performs model inference.

#### Responsibilities:

* Poll model from shared model store (hot reload)
* Use internal `StateBuilder` class to generate state from portfolio snapshot
* Request model prediction by passing the state to internal `ModelRunner` class
* Listen for inference results on Redis (or other message bus)
* Execute trades (or pass to ActionExecutor)

#### Components (Classes):

* `StateBuilder`: Constructs normalized state vector from live ticks + portfolio snapshot using **ta-lib or pandas-ta** (no custom indicator code)
* `ModelRunner`: Loads model, performs forward inference
* `InferenceAgent`: Coordinates flow between state, model, and output

#### Input:

* Portfolio snapshot (position, cash, inventory value, etc.)

#### Output:

* Executes model-pushed trade

---

### 2. **Trainer Module**

**Purpose:** Sole owner of learning. Responsible for generating model weights, computing reward, and managing experience replay.

#### Responsibilities:

* Listen to experience buffer for (s, a, r, s') tuples
* Perform online or batch DQN training
* Save updated model to model store
* Serve inference: upon state input, return model output (all heads)
* Compute reward using RewardAgent

#### Input:

* States and experiences from Core

#### Output:

* Updated model (to model store)
* Inference result (to Redis queue for Inference)

---

### 3. **Core Module**

**Purpose:** Houses all shared utilities, data preparation logic, and data bootstrap tools.

#### Responsibilities:

* Build state from portfolio snapshot (if invoked externally)
* Manage tick buffer (or fetch from Redis/Kafka)
* Compute indicators and feature transformations using **ta-lib or pandas-ta**
* Normalize state as per model\_config
* Maintain `model_config.yaml` and shared utility functions

#### Shared Utilities:

* Redis interface wrapper
* Model schema loader
* Feature scaling
* Logging and config loader
* Constants and enums

---

### 4. **UI Module**

**Purpose:** Real-time visualization and diagnostics of agent performance.

#### Responsibilities:

* Visualize trades, portfolio equity curve
* Show live model outputs: signal, confidence, reason weights
* Replay past state-action transitions
* Render current state vector, portfolio stats
* Display model version, strategy, cooldowns, stop-loss/take-profit levels

#### Data Sources:

* Redis (signal\_history, equity, experience logs)
* Model metadata from model store
* Optional: Snapshot from Core

---

## 🔄 System Flow

### Cold Start:

1. Core bootstraps historical OHLCV
2. Fills buffer → sends experience to Trainer
3. Trainer creates initial model → saves to store
4. Inference waits for model → becomes active

### Live Loop:

1. Inference builds state using `StateBuilder`
2. Inference passes state to `ModelRunner`
3. Trainer runs model(state) → sends result to Redis
4. Inference consumes and executes action
5. Core logs (s, a, s') to experience queue
6. Trainer consumes experience, computes reward, and updates model

---

## 🔐 Key Design Principles

* 🔁 Stateless inference — zero learning or state tracking logic
* 🧠 Centralized learning in Trainer only
* 📦 Minimal shared code in Core, all utility-based
* 📈 UI is read-only, pulls from queues and store
* ⚙️ All modules communicate via Redis or class interfaces
* 🧮 All indicators are computed using **ta-lib or pandas-ta**, no custom indicator implementations

---

## 📁 Updated Project Structure

```
rltrader/
├── build.log
├── deploy_rltrader_k3s.sh
├── docs
│   ├── Architecture.md
│   ├── dynamic_qnetwork_schema.md
│   ├── misc.yaml
│   └── model_config.yaml
├── k3s
│   ├── inference.yaml
│   ├── redis.yaml
│   ├── trainer.yaml
│   └── ui.yaml
├── push_all_to_local_registry.sh
└── src
    ├── core
    │   ├── config
    │   │   └── __init__.py
    │   ├── decorators
    │   │   ├── decorators.py
    │   │   └── __init__.py
    │   ├── __init__.py
    │   ├── requirements.txt
    │   └── utils
    │       ├── config_loader.py
    │       └── __init__.py
    ├── inference
    │   ├── agent
    │   │   └── __init__.py
    │   ├── Dockerfile
    │   ├── executor
    │   │   └── __init__.py
    │   ├── __init__.py
    │   └── requirements.txt
    ├── __init__.py
    ├── trainer
    │   ├── agent
    │   │   └── __init__.py
    │   ├── bootstrap
    │   │   └── __init__.py
    │   ├── Dockerfile
    │   ├── __init__.py
    │   └── requirements.txt
    └── ui
        ├── dashboard
        │   └── __init__.py
        ├── Dockerfile
        ├── __init__.py
        └── requirements.txt
```

