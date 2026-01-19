# Music Recommendation Engine

This project is a demo music recommendation engine that uses **K-Means clustering**
to group and recommend songs based on numerical audio features.  
It addresses the **cold-start problem** by relying on feature-based similarity
rather than user listening history.

## Features
- Feature-based music recommendations
- K-Means clustering for grouping similar songs
- Cold-start friendly approach
- Frontend + backend working together
- Modular design, easy to extend

## Tech Stack
- Frontend: Angular
- Backend: Python (Django)
- Machine Learning: Python, scikit-learn
- Data Processing: NumPy, Pandas

## Project Structure
music-app/
├── front-end/
├── backend/
├── ml/
└── README.md

## Machine Learning Details
- Run `ml/src/inference.py` to generate song recommendations
- Includes statistics about features, cluster analysis, and ideal number of clusters (k)
- Comparison between K-Means and DBSCAN:
  - **K-Means** produced 2 clusters (upbeat / downbeat)
  - **DBSCAN** classified ~20k songs as noise out of a 200k-song dataset
- Chose K-Means for stable clustering and recommendations despite DBSCAN’s noise issue

## How to Run
1. Start the backend server
2. Start the frontend server
3. Optionally, run `ml/src/inference.py` for ML recommendations
4. Access the application through the frontend interface

> Both frontend and backend must be running for the system to work properly.

## Future Improvements
- Add more audio features
- Improve recommendation quality
- Support user-based personalization
- Deploy the application

## Disclaimer
This project is for demonstration and educational purposes.
