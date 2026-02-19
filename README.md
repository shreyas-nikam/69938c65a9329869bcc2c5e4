Here's a comprehensive `README.md` file for your Streamlit application lab project, formatted in Markdown:

---

# QuLab: Lab 56: On-chain Anomaly Detection

<p align="center">
  <img src="https://www.quantuniversity.com/assets/img/logo5.jpg" alt="QuantUniversity Logo" width="200"/>
</p>

## Project Description

In the rapidly evolving and often opaque world of cryptocurrency, traditional financial analysis tools frequently fall short. **QuLab: Lab 56: On-chain Anomaly Detection** is a Streamlit application designed to equip financial professionals, particularly CFAs and Alternative Investments Analysts, with a robust, AI-driven methodology for navigating this complex landscape.

The project addresses the critical need to identify suspicious activities—such as massive "whale" movements, potential exchange hacks, or market manipulation through "wash trading"—*before* they impact portfolio performance or lead to regulatory scrutiny. This application provides a hands-on lab experience, guiding users through the end-to-end process of building and evaluating a real-time system to detect dramatic behavioral shifts in blockchain transaction patterns. Our ultimate goal is to generate actionable intelligence, functioning as "crypto-native fundamental analysis," to enable proactive risk management and informed decision-making in the challenging and high-velocity crypto market.

## Features

This application offers a guided, step-by-step workflow for understanding and implementing on-chain anomaly detection:

1.  **Overview**: An introductory section explaining the project's motivation, the CFA's challenge in crypto markets, and a high-level summary of the application's workflow.
2.  **Simulate Data**: Generate a synthetic dataset of blockchain transactions, meticulously embedding specific patterns for "normal" activity alongside various anomaly types: whale movements, hack attempts, and wash trading, providing a controlled environment for model development and validation.
3.  **Engineer Features**: Transform raw transaction data into 'crypto-native' features. This involves creating metrics that capture unique blockchain dynamics, such as log-transformed amounts and dormancy periods, Z-scores for statistical significance, and binary flags for specific behaviors (e.g., transactions to exchanges, new senders).
4.  **Detect Anomalies**: Implement a two-layer anomaly detection system. The first layer employs statistical Z-score alerts for immediate, univariate extremes. The second layer utilizes an **Isolation Forest** machine learning model to identify more complex, multivariate anomalies based on engineered features.
5.  **Classify Anomalies**: Apply rule-based logic to categorize the detected anomalies into specific, actionable types such as `WHALE_MOVEMENT`, `POSSIBLE_HACK`, `WASH_TRADING_SUSPECT`, and `LARGE_TRANSFER`. This step translates raw detections into meaningful intelligence for risk managers and compliance teams.
6.  **Market Impact Analysis**: Assess the potential market reaction to different anomaly types by analyzing simulated price changes following each event. This section demonstrates how on-chain signals can serve as leading indicators for market movements, providing a crucial edge for portfolio adjustment.
7.  **Monitoring Dashboard**: Define and display a clear, automated alert protocol. This protocol outlines the severity, expected response time, relevant notification parties (e.g., Trading Desk, Risk Manager, Compliance), and concrete actions required for each classified anomaly type, operationalizing risk management for 24/7 crypto market vigilance.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.7 or higher
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/QuantUniversity/QuLab-OnChain-Anomaly-Detection.git
    cd QuLab-OnChain-Anomaly-Detection
    ```
    *(Note: Replace `https://github.com/QuantUniversity/QuLab-OnChain-Anomaly-Detection.git` with the actual repository URL if it exists.)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required packages:**
    Create a `requirements.txt` file in the root directory of your project with the following content:
    ```
    streamlit>=1.0.0
    numpy
    pandas
    matplotlib
    seaborn
    scikit-learn
    ```
    Then, install the dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit application:

1.  Ensure you have completed the installation steps and activated your virtual environment.
2.  From the project's root directory (where `app.py` is located), execute the following command:
    ```bash
    streamlit run app.py
    ```

3.  The application will automatically open in your default web browser. If it doesn't, a local URL will be provided in your terminal (e.g., `http://localhost:8501`).
4.  Navigate through the different stages of the on-chain anomaly detection workflow using the sidebar menu on the left, starting from "Overview" and proceeding sequentially through the numbered steps.

## Project Structure

```
├── README.md                 # This documentation file.
├── app.py                    # The main Streamlit application code, handling UI and workflow logic.
├── source.py                 # Contains helper functions for data generation, feature engineering,
│                             # anomaly detection models, classification rules, and analysis logic.
└── requirements.txt          # Lists all Python dependencies required for the project.
```

## Technology Stack

*   **Python**: The foundational programming language for the entire application.
*   **Streamlit**: Utilized for building the interactive, user-friendly web application interface, enabling rapid prototyping and deployment of data science projects.
*   **NumPy**: Essential for high-performance numerical operations, particularly for array manipulations.
*   **Pandas**: A cornerstone for data manipulation and analysis, primarily used for handling and processing blockchain transaction data in DataFrames.
*   **Matplotlib**: A comprehensive library for creating static, animated, and interactive visualizations, used for plotting data distributions and anomaly timelines.
*   **Seaborn**: Built on Matplotlib, it provides a high-level interface for drawing attractive and informative statistical graphics, enhancing the visual presentation of data.
*   **Scikit-learn**: A robust machine learning library providing efficient tools for data preprocessing (e.g., `StandardScaler`) and unsupervised learning algorithms (e.g., `IsolationForest` for anomaly detection).

## Contributing

Contributions are highly welcome! If you have suggestions for improvements, new features, bug fixes, or want to enhance the analysis, please follow these guidelines:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bugfix (e.g., `git checkout -b feature/add-new-metric` or `bugfix/fix-classification-error`).
3.  **Make your changes**, ensuring your code adheres to a consistent style and includes appropriate comments.
4.  **Commit your changes** with clear and concise messages (e.g., `git commit -m 'feat: Added new feature for X'`).
5.  **Push your branch** to your forked repository (e.g., `git push origin feature/add-new-metric`).
6.  **Open a Pull Request** against the main repository, providing a detailed description of your changes and their purpose.

## License

This project is licensed under the MIT License - see the `LICENSE` file (if available in the repository) for details.

## Contact

For any questions, feedback, or further information regarding this project or QuantUniversity's programs, please feel free to reach out:

*   **Organization**: QuantUniversity
*   **Website**: [https://www.quantuniversity.com](https://www.quantuniversity.com)
*   **Email**: info@quantuniversity.com

---