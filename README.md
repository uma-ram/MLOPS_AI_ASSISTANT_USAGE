# MLOPS_AI_ASSISTANT_USAGE
This is MLOPS Zoomcamp's final project

MLOPS_AI_ASSISTANT_USAGE/
├── data/                        # Store raw and processed data
│   └── ai_assistant.csv
├── notebooks/                  # EDA and experimentation
│   └── eda.ipynb
├── scripts/                    # Modular Python scripts
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/                     # Saved model artifacts
├── mlruns/                     # MLflow tracking directory
├── prefect/                    # Prefect flows and deployment scripts
│   └── flow.py
├── tests/                      # Unit and integration tests
├── Dockerfile
├── requirements.txt
├── .pre-commit-config.yaml
├── .github/                    # GitHub Actions for CI/CD
│   └── workflows/
│       └── ci.yml
├── main.py                     # FastAPI entrypoint
├── Makefile
└── README.md
