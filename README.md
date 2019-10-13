# Script Breakdown

These python scripts form part of the analysis for my thesis. They perform intra-subreddit diversity analysis, a previous iteration of which was published in my article ['(Anti-)Echo Chamber Participation
Examining Contributor Activity Beyond the Chamber'](https://ellaguest.com/files/guest-antiechochambers-wp-2018.pdf).

## collection.py
- Collects all data from GCS BigQuery, stores in GCS storage buckets
- corresponds with Chapter 3: Data & Methodology

## descriptives.py
- Runs subreddit-level descriptive statistics
- corresponds with Chapter 4: Analysis 1 - Subreddit Level Echo Chambers

## tools.py
- Includes general tool functions to be called by other scripts 