import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

class TransactionRiskAnalyzer:
    HIGH_RISK_COUNTRIES: List[str] = [
        'AL', 'PK', 'UA', 'RU', 'NG', 'AF', 'DZ', 'AS', 'AO', 'AI', 'AG', 'AM', 'AZ', 'BS', 'BD', 'BB', 'BY', 'BZ', 
        'BJ', 'BO', 'BA', 'BW', 'BV', 'BG', 'BF', 'MM', 'BI', 'CV', 'KH', 'CM', 'KY', 'CF', 'TD', 'CN', 'CO', 'KM', 
        'CD', 'CG', 'CR', 'CI', 'HR', 'CU', 'CW', 'CY', 'DJ', 'DM', 'DO', 'TP', 'EC', 'EG', 'SV', 'GQ', 'ER', 'SZ'
    ]
    
    RISK_WEIGHTS: Dict[str, float] = {
        'freq_transaction_flag': 2.0,
        'high_value_static_flag': 1.5,
        'high_value_dynamic_flag': 1.0,
        'cross_border_flag': 2.5,
        'self_transfer_flag': 1.0,
        'multiple_device_flag': 1.5,
        'frequent_recipient_flag': 1.8,
        'round_amount_flag': 1.0,
        'high_risk_country_flag': 2.5,
        'monthly_volume_flag': 1.5,
        'behavior_change_flag': 1.8
    }
    
    RISK_THRESHOLDS: Dict[str, Tuple[float, float]] = {
        'Low': (0, 3),
        'Medium': (3, 5),
        'High': (5, 8),
        'Very High': (8, float('inf'))
    }

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['payment_type'] = df['payment_type'].fillna('UNK')
        df['date_request_submitted'] = pd.to_datetime(df['date_request_submitted'], errors='coerce')
        return df

    @staticmethod
    def calculate_user_statistics(df: pd.DataFrame) -> pd.DataFrame:
        df['user_avg_invoice'] = df.groupby('user_id')['invoice_value'].transform('mean')
        df['user_std_invoice'] = df.groupby('user_id')['invoice_value'].transform('std')
        df['device_count'] = df.groupby('user_id')['device'].transform('nunique')
        df['recipient_count'] = df.groupby('user_id')['target_recipient_id'].transform('nunique')
        return df

    @staticmethod
    def calculate_flags(df: pd.DataFrame, high_value_threshold: float = 5000) -> pd.DataFrame:
        flags = {
            'freq_transaction_flag': df['days_since_previous_req'] < 1,
            'high_value_static_flag': df['invoice_value'] > high_value_threshold,
            'high_value_dynamic_flag': df['invoice_value'] > (df['user_avg_invoice'] + 2 * df['user_std_invoice']),
            'cross_border_flag': df['addr_country_code'] != df['recipient_country_code'],
            'self_transfer_flag': df['transfer_to_self'] == 'Exact name match',
            'multiple_device_flag': df['device_count'] > 3,
            'frequent_recipient_flag': df['recipient_count'] > 5,
            'round_amount_flag': df['invoice_value'] % 1000 == 0,
            'high_risk_country_flag': df.apply(
                lambda x: x['addr_country_code'] in TransactionRiskAnalyzer.HIGH_RISK_COUNTRIES or 
                         x['recipient_country_code'] in TransactionRiskAnalyzer.HIGH_RISK_COUNTRIES, 
                axis=1
            ),
            'behavior_change_flag': abs(df['invoice_value'] - df['user_avg_invoice']) > (3 * df['user_std_invoice'])
        }
        
        for flag_name, flag_values in flags.items():
            df[flag_name] = flag_values

        df['monthly_tx_volume'] = df.groupby(
            ['user_id', df['date_request_submitted'].dt.to_period('M')]
        )['invoice_value'].transform('sum')
        df['monthly_volume_flag'] = df['monthly_tx_volume'] > high_value_threshold * 10
        
        return df

    @staticmethod
    def calculate_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
        df['risk_score'] = sum(
            df[flag] * weight 
            for flag, weight in TransactionRiskAnalyzer.RISK_WEIGHTS.items()
        )
        
        df['risk_category'] = df['risk_score'].apply(
            lambda score: next(
                (level for level, (low, high) in TransactionRiskAnalyzer.RISK_THRESHOLDS.items() 
                 if low <= score < high), 
                'Unknown'
            )
        )
        return df

    @staticmethod
    def perform_clustering(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
        features = ['risk_score', 'monthly_tx_volume']
        X = df[features].fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['risk_segment'] = kmeans.fit_predict(X_scaled)
        
        return df

    @staticmethod
    def analyze_transactions(df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = TransactionRiskAnalyzer.preprocess_data(df)
            df = TransactionRiskAnalyzer.calculate_user_statistics(df)
            df = TransactionRiskAnalyzer.calculate_flags(df)
            df = TransactionRiskAnalyzer.calculate_risk_scores(df)
            df = TransactionRiskAnalyzer.perform_clustering(df)
            return df
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def plot_risk_distribution(df: pd.DataFrame):
        fig = go.Figure()
        
        # Risk Category Distribution
        fig.add_trace(go.Bar(
            x=df['risk_category'].value_counts().index,
            y=df['risk_category'].value_counts().values,
            name='Risk Category Count'
        ))
        
        # Risk Score Distribution
        fig.add_trace(go.Histogram(
            x=df['risk_score'],
            name='Risk Score Distribution',
            nbinsx=20,
            marker=dict(color='crimson', opacity=0.7)
        ))
        
        fig.update_layout(
            title="Risk Distribution Analysis",
            xaxis_title="Risk Score / Category",
            yaxis_title="Frequency",
            barmode='group'
        )
        
        fig.show()
        
        # Monthly Transaction Volume by Risk Category with Custom Hover Data
        df['month'] = df['date_request_submitted'].dt.to_period('M')
        monthly_volume_by_category = df.groupby(['month', 'risk_category'])['invoice_value'].sum().reset_index()
        
        fig2 = go.Figure()
        
        for category in df['risk_category'].unique():
            category_data = monthly_volume_by_category[monthly_volume_by_category['risk_category'] == category]
            fig2.add_trace(go.Scatter(
                x=category_data['month'].astype(str),
                y=category_data['invoice_value'],
                mode='lines+markers',
                name=f"{category} Category",
                hoverinfo='text',
                text=[f"User ID: {row['user_id']}<br>Payment Type: {row['payment_type']}<br>Counterparty Country: {row['recipient_country_code']}" 
                      for _, row in df[df['risk_category'] == category].iterrows()]  # Tooltip information
            ))
        
        fig2.update_layout(
            title="Monthly Transaction Volume by Risk Category",
            xaxis_title="Month",
            yaxis_title="Transaction Volume",
            showlegend=True
        )
        
        fig2.show()
        
        # Daily Transaction Volume by Risk Category with Custom Hover Data
        df['day'] = df['date_request_submitted'].dt.to_period('D')
        daily_volume_by_category = df.groupby(['day', 'risk_category'])['invoice_value'].sum().reset_index()
        
        fig3 = go.Figure()
        
        for category in df['risk_category'].unique():
            category_data = daily_volume_by_category[daily_volume_by_category['risk_category'] == category]
            fig3.add_trace(go.Scatter(
                x=category_data['day'].astype(str),
                y=category_data['invoice_value'],
                mode='lines+markers',
                name=f"{category} Category",
                hoverinfo='text',
                text=[f"User ID: {row['user_id']}<br>Payment Type: {row['payment_type']}<br>Counterparty Country: {row['recipient_country_code']}" 
                      for _, row in df[df['risk_category'] == category].iterrows()]  # Tooltip information
            ))
        
        fig3.update_layout(
            title="Daily Transaction Volume by Risk Category",
            xaxis_title="Day",
            yaxis_title="Transaction Volume",
            showlegend=True
        )
        
        fig3.show()

def main():
    try:
                # Load transaction data
        df = pd.read_csv('Servicing - Product homework dataset.csv')
        print(f"Loaded {len(df)} transactions")
        
        # Perform analysis
        results = TransactionRiskAnalyzer.analyze_transactions(df)
        
        if not results.empty:
            # Display summary statistics
            flag_counts = results[list(TransactionRiskAnalyzer.RISK_WEIGHTS.keys())].sum()
            print("\nSuspicious Transaction Flags Summary:")
            print(flag_counts)
            
            print("\nRisk Distribution:")
            print(results['risk_category'].value_counts())
            
            print("\nSample Risk Analysis:")
            cols_to_show = ['user_id', 'risk_score', 'risk_category', 'risk_segment']
            print(results[cols_to_show].head())
            
            # Export results to a CSV file
            output_file = 'risk_analysis_results.csv'
            results.to_csv(output_file, index=False)
            print(f"\nResults exported to '{output_file}'")
            
            # Plot the risk distribution and transaction volumes
            TransactionRiskAnalyzer.plot_risk_distribution(results)
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == '__main__':
    main()

       
