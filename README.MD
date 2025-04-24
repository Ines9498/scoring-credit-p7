# Scoring Crédit - Prêt à Dépenser

Ce projet vise à prédire la probabilité de défaut de clients à l'aide d'un modèle de machine learning.

## Structure :
- `/api/` : API FastAPI pour les prédictions
- `/dashboard/` : Interface Streamlit pour visualisation & explications SHAP

## Lancement en local :
### API :
```bash
cd api
uvicorn app:app --reload --port 8001
