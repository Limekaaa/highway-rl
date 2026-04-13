# RL Highway

Projet de renforcement pour la conduite autonome sur highway-env, avec comparaison de plusieurs approches d'exploration et d'observation.

## 1. Objectif du projet

Ce dépôt explore des agents RL dans un environnement de type autoroute (highway-env), avec une progression en trois étapes:

- Baselines simples: agent aléatoire et agent inactif.
- DQN maison sur observations cinématiques (type Kinematics).
- DQN avec politique CNN (Stable-Baselines3) sur observations image en niveaux de gris (GrayscaleObservation).

L'objectif est de comparer la qualité des politiques, leur robustesse et leurs profils d'apprentissage selon la représentation d'état utilisée.

## 2. Stratégies d'exploration et agents testés

### Baselines

- Random Agent: choisit des actions aléatoires, sert de référence basse.
- Idle Agent: comportement quasi passif, sert de second point de comparaison simple.

Ces deux agents sont évalués dans le notebook d'analyse DQN pour établir un niveau minimal de performance avant apprentissage.

### DQN (observation cinématique)

Implémentation custom PyTorch:

- Espace d'observation: Kinematics (positions/vitesses des véhicules proches).
- Politique d'exploration: epsilon-greedy décroissante.
- Stabilité: replay buffer + target network.

Ce pipeline est codé dans:

- highway/models/dqn/dqn.py
- highway/models/dqn/train.py
- highway/models/dqn/config.py

### DQN CNN (observation grayscale)

Implémentation basée sur Stable-Baselines3:

- Politique: CnnPolicy.
- Observation: GrayscaleObservation (frames empilées, redimensionnées).
- Évaluation périodique: EvalCallback avec sauvegarde du meilleur modèle.

Ce pipeline est codé dans:

- highway/models/cnn/train.py
- highway/models/cnn/config.py
- shared_core_config.py (CNN_TEST_CONFIG, CNN_EVAL_CONFIG)

## 3. Environnements et configurations

La création d'environnement est centralisée dans highway/scripts/environment.py via ConfigType:

- SHARED_CORE: configuration de base commune.
- TEST: configuration utilisée pour l'entraînement/évaluation DQN cinématique.
- TEST_CNN: configuration d'entraînement pour DQN CNN.
- EVAL_CNN: configuration d'évaluation pour DQN CNN.

Les paramètres principaux sont définis dans:

- shared_core_config.py

## 4. Installation rapide

Depuis la racine du projet:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 5. Commandes importantes

### 5.1 Lancer un nouvel entraînement

#### DQN cinématique (implémentation custom)

```powershell
python -m highway.models.dqn.train
```

Sorties générées:

- Modèle best: model_weights/dqn/dqn_best_model_<timestamp>.pth
- Checkpoints: model_weights/dqn/dqn_checkpoint_<timestamp>_ep...pth
- Courbes numpy: results/dqn/loss, results/dqn/reward, results/dqn/length
- Logs texte: logs/dqn/DQN_Training_<timestamp>.log

#### DQN CNN (Stable-Baselines3)

```powershell
python -m highway.models.cnn.train
```

Sorties générées (dans output-root):

- Meilleur modèle: best_model/best_model.zip
- Logs d'évaluation: eval_logs/evaluations.npz
- Courbes compactes: eval_curves.npz
- TensorBoard: tb/
- Modèle final: model.zip
- Config d'évaluation exportée: eval_config.json

### 5.2 Consulter les résultats

#### Option A: Notebooks d'analyse

- dqn_analysis.ipynb: baselines, chargement des modèles DQN, distributions reward/length, intervalles de confiance.
- cnn_analysis.ipynb: analyse dédiée au pipeline CNN.
- Les fonctions d'analyse (plotting, statistiques, chargement d'artefacts, wrapper agent SB3) sont importées depuis `highway/scripts/utils`.
- Pour l'analyse CNN en notebook: privilégier `SB3DQN.load(..., device="cpu", custom_objects={"buffer_size": 1})` pour limiter la mémoire, et utiliser `make_deep_copy=False` dans `run_one_episode`/`eval_agent`.

#### Option B: TensorBoard pour DQN CNN

```powershell
tensorboard --logdir outputs_cnn_dqn/tb
```

## 6. Structure du dépôt

Arborescence logique:

- highway/
	- models/
		- dqn/: DQN custom (réseau MLP, buffer, entraînement)
		- cnn/: DQN SB3 avec CnnPolicy
		- random_agent/ et idle_agent/: baselines
	- scripts/: création d'environnement, exécution et évaluation d'un agent
		- utils/: fonctions partagées pour notebooks (`plotting.py`, `statistics.py`, `paths.py`, `agents.py`)
- shared_core_config.py: configurations d'environnement partagées et variantes CNN
- dqn_analysis.ipynb / cnn_analysis.ipynb: analyses expérimentales
- requirements.txt: dépendances Python

## 7. Reproductibilité

- Les seeds et configurations sont centralisées dans les scripts et fichiers de config.
- Pour comparer des runs, conservez le timestamp des modèles et les fichiers .npy/.npz associés.
- Préférez des évaluations multi-seeds (déjà présent dans dqn_analysis.ipynb) pour éviter les conclusions sur un seul épisode.


# Project members
- [Alexandre Faure](https://github.com/alexandre-faure) : alexandre.faure@student-cs.fr)
- [Corentin Lasne](https://github.com/corentinlasne/) : corentin.lasne@student-cs.fr
- [Maxime Hanus](https://github.com/Limekaaa/) : maxime.hanus@student-cs.fr
- [Charles Croon](https://github.com/Ccroon17) : charles.croon@student-cs.fr