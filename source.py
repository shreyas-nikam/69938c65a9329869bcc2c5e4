import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set a random seed for reproducibility
np.random.seed(42)

def generate_blockchain_data(n_normal=50000, n_whale=20, n_hack=15, n_wash=15):
    """
    Generates a synthetic blockchain transaction dataset with injected normal and anomalous events.

    Parameters:
    n_normal (int): Number of normal transactions.
    n_whale (int): Number of whale movement anomalies.
    n_hack (int): Number of hack pattern anomalies.
    n_wash (int): Number of wash trading anomalies.

    Returns:
    pd.DataFrame: A DataFrame containing simulated blockchain transactions.
    """
    # Normal transactions
    normal = pd.DataFrame({
        'tx_hash': [f'0x{np.random.randint(1e15):016x}' for _ in range(n_normal)],
        'amount_btc': np.random.lognormal(np.log(0.5), 1.5, n_normal),
        'from_addr': [f'addr_{np.random.randint(10000):05d}' for _ in range(n_normal)],
        'to_addr': [f'addr_{np.random.randint(10000):05d}' for _ in range(n_normal)],
        'timestamp': pd.date_range('2024-01-01', periods=n_normal, freq='2min'),
        'block_height': np.arange(n_normal) // 5 + 800000,
        'is_to_exchange': np.random.binomial(1, 0.15, n_normal),
        'from_dormancy_days': np.random.exponential(30, n_normal),
        'anomaly_type': 'normal'
    })

    anomalies = []
    base_time = pd.Timestamp('2024-03-15')

    # Type 1: WHALE MOVEMENTS (large transfers from dormant wallets to exchanges)
    for i in range(n_whale):
        anomalies.append({
            'tx_hash': f'0xwhale{i:04d}',
            'amount_btc': np.random.uniform(500, 5000),
            'from_addr': f'whale_{i:03d}',
            'to_addr': f'exchange_{np.random.randint(5)}',
            'timestamp': base_time + pd.Timedelta(days=i*5),
            'block_height': 810000 + i*500,
            'is_to_exchange': 1,
            'from_dormancy_days': np.random.uniform(180, 1000),
            'anomaly_type': 'whale'
        })

    # Type 2: HACK PATTERNS (rapid small transfers from exchange hot wallet to unknown addresses)
    exchange_hot_wallet_addr = 'exchange_hot_wallet'
    for i in range(n_hack):
        anomalies.append({
            'tx_hash': f'0xhack{i:04d}',
            'amount_btc': np.random.uniform(10, 50),
            'from_addr': exchange_hot_wallet_addr,
            'to_addr': f'unknown_{np.random.randint(1000):04d}',
            'timestamp': base_time + pd.Timedelta(hours=i*0.5),
            'block_height': 810000 + i*10,
            'is_to_exchange': 0,
            'from_dormancy_days': 0,
            'anomaly_type': 'hack'
        })

    # Type 3: WASH TRADING (circular transactions between two specific addresses)
    for i in range(n_wash // 2):
        addr_a = f'wash_a_{i:03d}'
        addr_b = f'wash_b_{i:03d}'

        # Tx A -> B
        anomalies.append({
            'tx_hash': f'0xwash{i:04d}a',
            'amount_btc': np.random.uniform(50, 200),
            'from_addr': addr_a,
            'to_addr': addr_b,
            'timestamp': base_time + pd.Timedelta(days=i*2, minutes=np.random.randint(60)),
            'block_height': 810000 + i*100,
            'is_to_exchange': 0,
            'from_dormancy_days': np.random.uniform(0, 2),
            'anomaly_type': 'wash'
        })
        # Tx B -> A (reciprocal)
        anomalies.append({
            'tx_hash': f'0xwash{i:04d}b',
            'amount_btc': np.random.uniform(50, 200),
            'from_addr': addr_b,
            'to_addr': addr_a,
            'timestamp': base_time + pd.Timedelta(days=i*2, minutes=np.random.randint(60) + 1),
            'block_height': 810000 + i*100 + 1,
            'is_to_exchange': 0,
            'from_dormancy_days': np.random.uniform(0, 2),
            'anomaly_type': 'wash'
        })

    anom_df = pd.DataFrame(anomalies)
    df = pd.concat([normal, anom_df], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    return df

def engineer_onchain_features(df):
    """
    Creates crypto-specific features from raw transaction data.
    These features are designed to distinguish anomalous from normal transactions.

    Parameters:
    df (pd.DataFrame): DataFrame with raw blockchain transaction data.

    Returns:
    pd.DataFrame: DataFrame with engineered features.
    """
    df_copy = df.copy()

    # Amount-related features
    df_copy['amount_log'] = np.log1p(df_copy['amount_btc'])
    df_copy['amount_zscore'] = (df_copy['amount_btc'] - df_copy['amount_btc'].mean()) / df_copy['amount_btc'].std()
    df_copy['is_large'] = (df_copy['amount_btc'] > df_copy['amount_btc'].quantile(0.99)).astype(int)

    # Dormancy-related features
    df_copy['dormancy_log'] = np.log1p(df_copy['from_dormancy_days'])
    df_copy['is_dormant_sender'] = (df_copy['from_dormancy_days'] > 180).astype(int)

    # Address behavior features (simplified)
    from_counts = df_copy['from_addr'].value_counts()
    df_copy['sender_tx_count'] = df_copy['from_addr'].map(from_counts)

    to_counts = df_copy['to_addr'].value_counts()
    df_copy['receiver_tx_count'] = df_copy['to_addr'].map(to_counts)

    df_copy['is_new_sender'] = (df_copy['sender_tx_count'] <= 2).astype(int)

    # Counterparty concentration / unknown recipient
    df_copy['is_to_unknown'] = (~df_copy['to_addr'].str.startswith('addr_') &
                                 ~df_copy['to_addr'].str.startswith('exchange_')).astype(int)

    return df_copy

def zscore_alerts(df, amount_threshold=4.0, dormancy_threshold=3.0):
    """
    Generates statistical Z-score alerts for extreme univariate values.

    Parameters:
    df (pd.DataFrame): DataFrame with 'amount_btc' and 'from_dormancy_days' (or their Z-scores).
    amount_threshold (float): Z-score threshold for amount.
    dormancy_threshold (float): Z-score threshold for dormancy.

    Returns:
    pd.DataFrame: DataFrame with binary alert flags.
    """
    alerts = pd.DataFrame(index=df.index)
    alerts['amount_alert'] = df['amount_zscore'].abs() > amount_threshold

    dormancy_mean = df['from_dormancy_days'].mean()
    dormancy_std = df['from_dormancy_days'].std()
    dormancy_z = (df['from_dormancy_days'] - dormancy_mean) / dormancy_std
    alerts['dormancy_alert'] = dormancy_z.abs() > dormancy_threshold

    alerts['any_zscore_alert'] = alerts['amount_alert'] | alerts['dormancy_alert']
    return alerts

def apply_isolation_forest(df, feature_cols, contamination=0.005, random_state=42, n_estimators=200):
    """
    Applies Isolation Forest for multivariate anomaly detection.

    Parameters:
    df (pd.DataFrame): DataFrame with engineered features.
    feature_cols (list): List of feature columns to use for Isolation Forest.
    contamination (float): The proportion of outliers in the data set.
    random_state (int): Seed for reproducibility.
    n_estimators (int): The number of base estimators in the ensemble.

    Returns:
    pd.DataFrame: DataFrame with Isolation Forest anomaly score and flag.
    """
    df_copy = df.copy()
    X = StandardScaler().fit_transform(df_copy[feature_cols].fillna(0))

    iso_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state, n_jobs=-1)
    iso_forest.fit(X)

    df_copy['iso_forest_anomaly_score'] = iso_forest.decision_function(X)
    df_copy['is_flagged_iso_forest'] = (iso_forest.predict(X) == -1).astype(int)
    return df_copy

def perform_anomaly_detection(df_engineered, feature_cols, amount_threshold=4.0, dormancy_threshold=3.0, iso_forest_contamination=0.005, iso_forest_n_estimators=200):
    """
    Orchestrates the two-layer anomaly detection process (Z-score + Isolation Forest).

    Parameters:
    df_engineered (pd.DataFrame): DataFrame with engineered features.
    feature_cols (list): List of feature columns for Isolation Forest.
    amount_threshold (float): Z-score threshold for amount.
    dormancy_threshold (float): Z-score threshold for dormancy.
    iso_forest_contamination (float): Contamination parameter for Isolation Forest.
    iso_forest_n_estimators (int): Number of estimators for Isolation Forest.

    Returns:
    pd.DataFrame: DataFrame with combined anomaly detection flags.
    """
    # Layer 1: Z-score alerts
    df_with_alerts = df_engineered.copy()
    zscore_results = zscore_alerts(df_with_alerts, amount_threshold=amount_threshold, dormancy_threshold=dormancy_threshold)
    df_with_alerts['any_zscore_alert'] = zscore_results['any_zscore_alert']

    # Layer 2: Isolation Forest
    df_with_alerts = apply_isolation_forest(df_with_alerts, feature_cols, contamination=iso_forest_contamination, n_estimators=iso_forest_n_estimators)

    # Combine alerts
    df_with_alerts['final_alert'] = (df_with_alerts['is_flagged_iso_forest'] == 1) | df_with_alerts['any_zscore_alert']
    return df_with_alerts

def classify_anomaly(row):
    """
    Rule-based classification of detected anomalies into specific types.

    Parameters:
    row (pd.Series): A row from the DataFrame containing engineered features and detection flags.

    Returns:
    str: The classified anomaly type.
    """
    if row['amount_btc'] > 100 and row['is_dormant_sender'] == 1 and row['is_to_exchange'] == 1:
        return 'WHALE_MOVEMENT'
    elif row['from_addr'] == 'exchange_hot_wallet' and row['is_to_unknown'] == 1:
        return 'POSSIBLE_HACK'
    elif row['sender_tx_count'] <= 3 and row['from_dormancy_days'] < 2:
        return 'WASH_TRADING_SUSPECT'
    elif row['amount_btc'] > 500:
        return 'LARGE_TRANSFER'
    else:
        return 'UNCLASSIFIED'

def classify_detected_anomalies(df_detected, verbose=True):
    """
    Applies rule-based classification to flagged anomalies.

    Parameters:
    df_detected (pd.DataFrame): DataFrame with anomaly detection flags.
    verbose (bool): Whether to print classification summary.

    Returns:
    pd.DataFrame: DataFrame of flagged transactions with classified types.
    """
    flagged_txns = df_detected[df_detected['final_alert']].copy()
    flagged_txns['classified_type'] = flagged_txns.apply(classify_anomaly, axis=1)

    if verbose:
        print("--- ANOMALY CLASSIFICATION ---")
        print("=" * 55)
        for ctype, count in flagged_txns['classified_type'].value_counts().items():
            print(f" {ctype: <25s}: {count}")
    return flagged_txns

def evaluate_detection_performance(df_detected, verbose=True):
    """
    Calculates overall and per-type recall/precision.

    Parameters:
    df_detected (pd.DataFrame): DataFrame with original anomaly_type and final_alert flags.
    verbose (bool): Whether to print performance metrics.

    Returns:
    dict: Dictionary containing overall recall, precision, and per-type metrics.
    """
    true_anomalies = (df_detected['anomaly_type'] != 'normal')
    flagged = df_detected['final_alert']

    tp = (flagged & true_anomalies).sum()
    fp = (flagged & ~true_anomalies).sum()
    fn = (~flagged & true_anomalies).sum()

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)

    if verbose:
        print("\n--- OVERALL DETECTION PERFORMANCE ---")
        print(f"True anomalies: {true_anomalies.sum()}")
        print(f"Flagged: {flagged.sum()} (TP={tp}, FP={fp}, FN={fn})")
        print(f"Recall: {recall:.2%} | Precision: {precision:.2%}")

    per_type_metrics = {}
    if verbose:
        print("\n--- PER-TYPE DETECTION PERFORMANCE (Recall against injected anomalies) ---")
    for atype_injected in ('whale', 'hack', 'wash'):
        total_injected_type = (df_detected['anomaly_type'] == atype_injected).sum()
        # Re-apply classify_anomaly for accurate type-specific recall against original anomalies
        tp_classified_correctly = ((df_detected['anomaly_type'] == atype_injected) &
                                   (df_detected['final_alert']) &
                                   (df_detected.apply(classify_anomaly, axis=1) == atype_injected.upper())).sum()

        recall_type = tp_classified_correctly / max(total_injected_type, 1)
        if verbose:
            print(f"\n{atype_injected.upper()} Detection Recall: {tp_classified_correctly}/{total_injected_type} ({recall_type:.2%})")
        per_type_metrics[atype_injected] = {'tp': tp_classified_correctly, 'total': total_injected_type, 'recall': recall_type}

    if verbose:
        print(f"\nOverall Anomaly Detection Recall: {recall:.2%} | Precision: {precision:.2%}")

    return {'overall_recall': recall, 'overall_precision': precision, 'per_type_metrics': per_type_metrics}

def plot_amount_distribution(df, anomalies_df):
    """
    Generates a histogram of log-transformed transaction amounts with anomalies highlighted.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(df['amount_log'], bins=50, kde=True, label='All Transactions', color='gray')
    sns.histplot(anomalies_df['amount_log'], bins=50, color='red', label='Injected Anomalies', alpha=0.6)
    plt.title('Log-transformed Transaction Amount Distribution with Anomalies')
    plt.xlabel('Log(1 + Amount BTC)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_anomaly_timeline(df_detected):
    """
    Generates a scatter plot of transaction timeline with injected and detected anomalies.
    """
    plt.figure(figsize=(15, 7))
    sns.scatterplot(x='timestamp', y='amount_log', data=df_detected[df_detected['anomaly_type'] == 'normal'],
                    color='gray', alpha=0.5, label='Normal Transactions', s=10)
    sns.scatterplot(x='timestamp', y='amount_log', data=df_detected[df_detected['anomaly_type'] == 'whale'],
                    color='blue', label='Whale Movements', s=50, marker='D')
    sns.scatterplot(x='timestamp', y='amount_log', data=df_detected[df_detected['anomaly_type'] == 'hack'],
                    color='red', label='Hack Patterns', s=50, marker='X')
    sns.scatterplot(x='timestamp', y='amount_log', data=df_detected[df_detected['anomaly_type'] == 'wash'],
                    color='orange', label='Wash Trading', s=50, marker='o')

    # Highlight final detected anomalies
    sns.scatterplot(x='timestamp', y='amount_log', data=df_detected[df_detected['final_alert']],
                    facecolors='none', edgecolors='black', linewidth=1.5, s=100, label='Detected Anomaly', zorder=5)

    plt.title('Transaction Timeline with Injected and Detected Anomalies')
    plt.xlabel('Timestamp')
    plt.ylabel('Log(1 + Amount BTC)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def market_impact_analysis(classified_anomaly_events=None, plot_impact=True, verbose=True):
    """
    Simulates and analyzes the market impact of different anomaly types.
    In a real scenario, this would involve joining with actual price data
    and calculating average price changes in specific windows.
    For this lab, we use simulated impacts based on common observations.

    Parameters:
    classified_anomaly_events (pd.DataFrame): DataFrame of detected and classified anomalies.
                                             (Not directly used for impact calculation in this sim,
                                             but conceptually represents the input).
    plot_impact (bool): Whether to generate the market impact bar chart.
    verbose (bool): Whether to print impact summary.
    """
    if verbose:
        print("--- MARKET IMPACT ANALYSIS ---")
        print("=" * 60)

    # Simulated average price impacts
    impacts = {
        'WHALE_MOVEMENT': {'1h': -0.8, '4h': -2.1, '24h': -3.5},
        'POSSIBLE_HACK': {'1h': -1.5, '4h': -5.2, '24h': -8.0},
        'WASH_TRADING_SUSPECT': {'1h': 0.1, '4h': 0.3, '24h': -0.5},
        'LARGE_TRANSFER': {'1h': -0.3, '4h': -0.8, '24h': -1.2},
        'UNCLASSIFIED': {'1h': 0.0, '4h': 0.0, '24h': 0.0}
    }

    if verbose:
        print(f"{'Event Type':<25s} {'1h Impact':>10s} {'4h Impact':>10s} {'24h Impact':>10s}")
        print("-" * 60)
        for etype, impact in impacts.items():
            print(f"{etype:<25s} {impact['1h']:>+10.1f}% {impact['4h']:>+10.1f}% {impact['24h']:>+10.1f}%")

        print("\nKey findings:")
        print(" - Whale movements precede an average -2.1% price drop within 4h.")
        print(" - Hack events precede an average -5.2% price drop within 4h, indicating severe market reaction.")
        print(" - Wash trading has minimal directional impact (noise, not a strong signal).")
        print("\nIMPLICATION: On-chain anomaly detection provides actionable alpha in crypto, not by predicting returns (which is very hard) but by detecting events that precede moves.")

    if plot_impact:
        impact_df = pd.DataFrame(impacts).T
        impact_df.index.name = 'Anomaly Type'
        melted_impact_df = impact_df.reset_index().melt(id_vars='Anomaly Type', var_name='Time Window', value_name='Avg Price Change (%)')

        plt.figure(figsize=(12, 7))
        sns.barplot(x='Anomaly Type', y='Avg Price Change (%)', hue='Time Window', data=melted_impact_df, palette='viridis')
        plt.title('Average BTC Price Change Following Different Anomaly Types')
        plt.ylabel('Average Price Change (%)')
        plt.xlabel('Anomaly Type')
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Time Window')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

def onchain_monitoring_dashboard(verbose=True):
    """
    Defines the specification for a real-time on-chain monitoring dashboard
    and its associated alert protocol.

    Parameters:
    verbose (bool): Whether to print dashboard specifications.

    Returns:
    dict: Dictionary specifying the monitoring alert protocol.
    """
    if verbose:
        print("--- ON-CHAIN MONITORING ALERT PROTOCOL ---")
        print("=" * 60)

    ALERT_PROTOCOL = {
        'WHALE_MOVEMENT': {
            'severity': 'HIGH',
            'response_time': '< 15 min',
            'notify': ['Trading desk', 'Risk manager'],
            'action': 'Review long exposure. Consider reducing position.'
        },
        'POSSIBLE_HACK': {
            'severity': 'CRITICAL',
            'response_time': '< 5 min',
            'notify': ['Trading desk', 'Risk manager', 'Compliance', 'CISO'],
            'action': 'Withdraw funds from affected exchange. Halt trading.'
        },
        'WASH_TRADING_SUSPECT': {
            'severity': 'MEDIUM',
            'response_time': '< 1 hour',
            'notify': ['Compliance', 'Market surveillance'],
            'action': 'Investigate. Exclude from volume analysis.'
        },
        'LARGE_TRANSFER': {
            'severity': 'MEDIUM',
            'response_time': '< 30 min',
            'notify': ['Trading desk', 'Risk manager'],
            'action': 'Review asset liquidity and exposure.'
        },
        'UNCLASSIFIED': {
            'severity': 'LOW',
            'response_time': '< 4 hours',
            'notify': ['Risk analyst'],
            'action': 'Further manual investigation of features.'
        }
    }

    if verbose:
        for alert_type, protocol in ALERT_PROTOCOL.items():
            print(f"\n{alert_type}:")
            for k, v in protocol.items():
                if isinstance(v, list):
                    print(f"  {k}: {', '.join(v)}")
                else:
                    print(f"  {k}: {v}")

        print("\n--- OPERATIONAL REQUIREMENTS ---")
        print("Coverage: 24/7 (crypto never sleeps)")
        print("Latency: < 30 seconds from block confirmation to alert")
        print("Data sources: Block explorer API (Etherscan, Blockchain.com)")
        print("Monitoring agent: AI agent from D3-T3 running continuously")

    return ALERT_PROTOCOL

def run_anomaly_detection_pipeline(
    n_normal=50000, n_whale=20, n_hack=15, n_wash=15,
    amount_zscore_threshold=4.0, dormancy_zscore_threshold=3.0,
    iso_forest_contamination=0.005, iso_forest_n_estimators=200,
    plot_results=True, verbose=True
):
    """
    Runs the complete blockchain anomaly detection pipeline.

    Parameters:
    n_normal (int): Number of normal transactions for data generation.
    n_whale (int): Number of whale movement anomalies.
    n_hack (int): Number of hack pattern anomalies.
    n_wash (int): Number of wash trading anomalies.
    amount_zscore_threshold (float): Z-score threshold for transaction amount.
    dormancy_zscore_threshold (float): Z-score threshold for sender dormancy.
    iso_forest_contamination (float): Contamination parameter for Isolation Forest.
    iso_forest_n_estimators (int): Number of estimators for Isolation Forest.
    plot_results (bool): Whether to display plots.
    verbose (bool): Whether to print detailed messages during execution.

    Returns:
    tuple: (txns_detected, flagged_txns, performance_metrics, alert_protocol)
        txns_detected (pd.DataFrame): DataFrame with all transactions and detection flags.
        flagged_txns (pd.DataFrame): DataFrame containing only the flagged and classified anomalies.
        performance_metrics (dict): Dictionary with overall and per-type detection metrics.
        alert_protocol (dict): Dictionary specifying the monitoring alert protocol.
    """
    if verbose: print("Starting blockchain anomaly detection pipeline...")

    # 1. Generate Data
    if verbose: print("\n1. Generating synthetic blockchain data...")
    txns = generate_blockchain_data(n_normal, n_whale, n_hack, n_wash)
    if verbose:
        print(f"Total transactions simulated: {len(txns)}")
        print(f"Number of anomalies injected: {(txns['anomaly_type'] != 'normal').sum()}")
        for atype in ('whale', 'hack', 'wash'):
            print(f"  - {atype.capitalize()} movements: {(txns['anomaly_type']==atype).sum()}")
        print("\nFirst 5 transactions:")
        print(txns.head())

    # 2. Engineer Features
    if verbose: print("\n2. Engineering on-chain features...")
    txns_engineered = engineer_onchain_features(txns)
    feature_cols = [
        'amount_log', 'amount_zscore', 'dormancy_log', 'is_large',
        'is_dormant_sender', 'is_to_exchange', 'sender_tx_count',
        'receiver_tx_count', 'is_new_sender', 'is_to_unknown'
    ]
    if verbose:
        print(f"On-chain features engineered: {len(feature_cols)}")
        print("First 5 transactions with new features:")
        print(txns_engineered[feature_cols + ['anomaly_type']].head())

    if plot_results:
        plot_amount_distribution(txns_engineered, txns_engineered[txns_engineered['anomaly_type'] != 'normal'])

    # 3. Perform Anomaly Detection
    if verbose: print("\n3. Performing two-layer anomaly detection (Z-score + Isolation Forest)...")
    txns_detected = perform_anomaly_detection(
        txns_engineered, feature_cols,
        amount_threshold=amount_zscore_threshold,
        dormancy_threshold=dormancy_zscore_threshold,
        iso_forest_contamination=iso_forest_contamination,
        iso_forest_n_estimators=iso_forest_n_estimators
    )
    if verbose:
        print(f"Transactions flagged by Z-score alerts: {txns_detected['any_zscore_alert'].sum()}")
        print(f"Transactions flagged by Isolation Forest: {txns_detected['is_flagged_iso_forest'].sum()}")
        print(f"Total transactions flagged by two-layer detection: {txns_detected['final_alert'].sum()}")

    # 4. Classify Detected Anomalies
    if verbose: print("\n4. Classifying detected anomalies...")
    flagged_txns = classify_detected_anomalies(txns_detected, verbose=verbose)

    # 5. Evaluate Detection Performance
    if verbose: print("\n5. Evaluating detection performance...")
    performance_metrics = evaluate_detection_performance(txns_detected, verbose=verbose)

    if plot_results:
        plot_anomaly_timeline(txns_detected)

    # 6. Market Impact Analysis
    if verbose: print("\n6. Performing market impact analysis (simulated)...")
    market_impact_analysis(flagged_txns, plot_impact=plot_results, verbose=verbose)

    # 7. Operational Monitoring Dashboard Specification
    if verbose: print("\n7. Defining operational monitoring dashboard and alert protocol...")
    alert_protocol = onchain_monitoring_dashboard(verbose=verbose)

    if verbose: print("\nBlockchain anomaly detection pipeline finished.")

    return txns_detected, flagged_txns, performance_metrics, alert_protocol
