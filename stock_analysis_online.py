import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="📈 Analyse Technique d'Actions",
    page_icon="📊",
    layout="wide"
)

# Titre principal
st.title("📈 Analyse Technique Complète d'Actions")
st.markdown("---")

# Sidebar pour la configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Input du ticker
    ticker = st.text_input("🎯 Ticker de l'action:", value="MU", help="Ex: AAPL, TSLA, GOOGL, MU").upper()
    
    # Période d'analyse
    period_options = {
        "1 an": 365,
        "2 ans": 730,
        "3 ans": 1095,
        "5 ans": 1825
    }
    period_label = st.selectbox("📅 Période d'analyse:", list(period_options.keys()), index=3)
    period_days = period_options[period_label]
    
    # Bouton d'analyse
    analyze_button = st.button("🚀 Lancer l'Analyse", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.info("💡 Entrez un ticker (ex: AAPL, TSLA, NVDA) et cliquez sur 'Lancer l'Analyse'")

# Fonction pour calculer les indicateurs techniques
def calculate_indicators(df):
    """Calcule tous les indicateurs techniques"""
    
    # Moyennes mobiles
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # Moyennes mobiles exponentielles
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    window = 20
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    df['BB_Std'] = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    
    # Volatilité
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
    
    # VWAP
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # Support et résistance
    df['Support'] = df['Low'].rolling(window=20).min()
    df['Resistance'] = df['High'].rolling(window=20).max()
    
    # Rendements
    df['Daily_Return'] = df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
    
    return df

# Fonction pour créer le graphique principal
def create_main_chart(df, ticker_symbol):
    """Crée le graphique principal avec prix, volume, MACD et RSI"""
    
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Prix et Moyennes Mobiles', 'Volume', 'MACD', 'RSI'),
        vertical_spacing=0.05,
        row_heights=[0.5, 0.2, 0.15, 0.15]
    )
    
    # Prix et moyennes mobiles
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Prix de clôture', 
                   line=dict(color='#00ff9f', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MA_20'], name='MA 20',
                   line=dict(color='#ffa500', width=1.5)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MA_50'], name='MA 50',
                   line=dict(color='#ff6b6b', width=1.5)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MA_200'], name='MA 200',
                   line=dict(color='#4ecdc4', width=2)),
        row=1, col=1
    )
    
    # Bandes de Bollinger
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Upper'], 
                   line=dict(color='rgba(255,255,255,0.3)', width=1),
                   name='BB Supérieure', showlegend=False),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Lower'], 
                   fill='tonexty', fillcolor='rgba(255,255,255,0.1)',
                   line=dict(color='rgba(255,255,255,0.3)', width=1),
                   name='Zone Bollinger', showlegend=True),
        row=1, col=1
    )
    
    # Volume
    colors = ['#ff4757' if df['Close'].iloc[i] < df['Open'].iloc[i] 
              else '#2ed573' for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume',
               marker_color=colors, opacity=0.6),
        row=2, col=1
    )
    
    # MACD
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                   line=dict(color='#3742fa', width=2)),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                   line=dict(color='#ff6348', width=1.5)),
        row=3, col=1
    )
    
    # Histogramme MACD
    hist_colors = ['#2ed573' if x >= 0 else '#ff4757' for x in df['MACD_Histogram']]
    fig.add_trace(
        go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogramme',
               marker_color=hist_colors, opacity=0.6),
        row=3, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                   line=dict(color='#ffa502', width=2)),
        row=4, col=1
    )
    
    # Lignes de référence RSI
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,255,255,0.5)", 
                  row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(255,255,255,0.5)", 
                  row=4, col=1)
    
    # Configuration du layout
    fig.update_layout(
        title=dict(
            text=f'<b>Analyse Technique Complète - {ticker_symbol}</b>',
            x=0.5,
            font=dict(size=24, color='white')
        ),
        template='plotly_dark',
        height=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    
    return fig

# Fonction pour créer le graphique de volatilité
def create_volatility_chart(df, ticker_symbol):
    """Crée le graphique d'analyse de volatilité"""
    
    fig_vol = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Volatilité dans le temps', 'Distribution des rendements quotidiens',
                       'Rendements cumulatifs', 'Corrélation Volume-Prix'),
        specs=[[{"secondary_y": True}, {"type": "histogram"}],
               [{"colspan": 1}, {"type": "scatter"}]]
    )
    
    # Volatilité
    fig_vol.add_trace(
        go.Scatter(x=df.index, y=df['Volatility'], name='Volatilité (%)',
                   line=dict(color='#ff6b6b', width=2)),
        row=1, col=1
    )
    
    # Prix sur axe secondaire
    fig_vol.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Prix',
                   line=dict(color='#4ecdc4', width=1, dash='dot')),
        row=1, col=1, secondary_y=True
    )
    
    # Distribution des rendements
    fig_vol.add_trace(
        go.Histogram(x=df['Daily_Return'].dropna()*100, nbinsx=50,
                    name='Rendements (%)', marker_color='#45aaf2',
                    opacity=0.7),
        row=1, col=2
    )
    
    # Rendements cumulatifs
    fig_vol.add_trace(
        go.Scatter(x=df.index, y=df['Cumulative_Return']*100, 
                   name='Rendement Cumulatif (%)',
                   line=dict(color='#26de81', width=3),
                   fill='tozeroy', fillcolor='rgba(38, 222, 129, 0.1)'),
        row=2, col=1
    )
    
    # Corrélation Volume-Prix
    price_change = df['Daily_Return'].dropna()
    volume_change = df['Volume'].pct_change().dropna()
    min_len = min(len(price_change), len(volume_change))
    price_change = price_change[-min_len:]
    volume_change = volume_change[-min_len:]
    
    fig_vol.add_trace(
        go.Scatter(x=volume_change*100, y=price_change*100,
                   mode='markers', name='Vol vs Prix',
                   marker=dict(color='#fd79a8', size=4, opacity=0.6)),
        row=2, col=2
    )
    
    # Configuration
    fig_vol.update_xaxes(title_text="Date", row=1, col=1)
    fig_vol.update_yaxes(title_text="Volatilité (%)", row=1, col=1)
    fig_vol.update_yaxes(title_text="Prix ($)", row=1, col=1, secondary_y=True)
    fig_vol.update_xaxes(title_text="Rendement quotidien (%)", row=1, col=2)
    fig_vol.update_yaxes(title_text="Fréquence", row=1, col=2)
    fig_vol.update_xaxes(title_text="Date", row=2, col=1)
    fig_vol.update_yaxes(title_text="Rendement Cumulatif (%)", row=2, col=1)
    fig_vol.update_xaxes(title_text="Variation Volume (%)", row=2, col=2)
    fig_vol.update_yaxes(title_text="Rendement Prix (%)", row=2, col=2)
    
    fig_vol.update_layout(
        title=dict(
            text=f'<b>Analyse de la Volatilité et des Rendements - {ticker_symbol}</b>',
            x=0.5,
            font=dict(size=20, color='white')
        ),
        template='plotly_dark',
        height=800,
        showlegend=True
    )
    
    return fig_vol

# Fonction pour créer le graphique support/résistance
def create_support_resistance_chart(df, ticker_symbol):
    """Crée le graphique d'analyse support/résistance"""
    
    fig_sr = go.Figure()
    
    # Données des 6 derniers mois
    recent_data = df.last('6M').copy()
    
    # Prix principal (Candlestick)
    fig_sr.add_trace(go.Candlestick(
        x=recent_data.index,
        open=recent_data['Open'],
        high=recent_data['High'],
        low=recent_data['Low'],
        close=recent_data['Close'],
        name='Prix',
        increasing_line_color='#26de81',
        decreasing_line_color='#ff4757'
    ))
    
    # Support et résistance
    fig_sr.add_trace(go.Scatter(
        x=recent_data.index, y=recent_data['Support'],
        name='Support', line=dict(color='#ff6b6b', width=2, dash='dash'),
        fill=None
    ))
    
    fig_sr.add_trace(go.Scatter(
        x=recent_data.index, y=recent_data['Resistance'],
        name='Résistance', line=dict(color='#4ecdc4', width=2, dash='dash'),
        fill='tonexty', fillcolor='rgba(255, 107, 107, 0.1)'
    ))
    
    # VWAP
    fig_sr.add_trace(go.Scatter(
        x=recent_data.index, y=recent_data['VWAP'],
        name='VWAP', line=dict(color='#ffa502', width=2)
    ))
    
    # Moyennes mobiles
    fig_sr.add_trace(go.Scatter(
        x=recent_data.index, y=recent_data['MA_20'],
        name='MA 20', line=dict(color='#fd79a8', width=1.5)
    ))
    
    fig_sr.add_trace(go.Scatter(
        x=recent_data.index, y=recent_data['MA_50'],
        name='MA 50', line=dict(color='#45aaf2', width=1.5)
    ))
    
    # Configuration
    fig_sr.update_layout(
        title=dict(
            text=f'<b>Analyse Support/Résistance - {ticker_symbol} (6 derniers mois)</b>',
            x=0.5,
            font=dict(size=20, color='white')
        ),
        template='plotly_dark',
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
    return fig_sr

# Fonction pour la modélisation prédictive
def create_prediction_model(df):
    """Crée et entraîne le modèle de prédiction"""
    
    # Création des features
    lookback_days = 5
    features_df = df.copy()
    
    # Features de prix
    for i in range(1, lookback_days + 1):
        features_df[f'Close_lag_{i}'] = features_df['Close'].shift(i)
        features_df[f'Volume_lag_{i}'] = features_df['Volume'].shift(i)
        features_df[f'Return_lag_{i}'] = features_df['Daily_Return'].shift(i)
    
    # Features techniques
    features_df['Price_MA20_ratio'] = features_df['Close'] / features_df['MA_20']
    features_df['Price_MA50_ratio'] = features_df['Close'] / features_df['MA_50']
    features_df['RSI_level'] = features_df['RSI']
    features_df['MACD_signal'] = (features_df['MACD'] > features_df['MACD_Signal']).astype(int)
    features_df['BB_position'] = (features_df['Close'] - features_df['BB_Lower']) / (features_df['BB_Upper'] - features_df['BB_Lower'])
    features_df['Vol_ratio'] = features_df['Volatility'] / features_df['Volatility'].mean()
    
    # Target
    features_df['Target'] = features_df['Close'].shift(-1) / features_df['Close'] - 1
    
    # Sélection des features
    feature_columns = [col for col in features_df.columns if 
                      ('lag_' in col or 'ratio' in col or 'level' in col or 
                       'signal' in col or 'position' in col or 'Vol_ratio' in col)]
    
    # Nettoyage
    ml_clean = features_df[feature_columns + ['Target']].dropna()
    
    # Division train/test
    split_point = int(len(ml_clean) * 0.8)
    train_data = ml_clean.iloc[:split_point]
    test_data = ml_clean.iloc[split_point:]
    
    X_train = train_data[feature_columns]
    y_train = train_data['Target']
    X_test = test_data[feature_columns]
    y_test = test_data['Target']
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Métriques
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return rf_model, feature_columns, test_data, rf_pred, rf_r2, rf_mse, feature_importance

# Main application logic
if analyze_button:
    try:
        with st.spinner(f'🔄 Récupération des données pour {ticker}...'):
            # Récupération des données
            stock = yf.Ticker(ticker)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                st.error(f"❌ Aucune donnée trouvée pour le ticker '{ticker}'. Vérifiez le symbole.")
                st.stop()
            
            # Informations sur l'entreprise
            info = stock.info
            
        st.success(f"✅ Données récupérées: {len(data)} jours de {data.index[0].strftime('%Y-%m-%d')} à {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Informations générales
        st.header(f"🏢 {info.get('longName', ticker)}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💼 Secteur", info.get('sector', 'N/A'))
        with col2:
            st.metric("🏭 Industrie", info.get('industry', 'N/A')[:20] + "...")
        with col3:
            market_cap = info.get('marketCap', 0)
            if market_cap > 0:
                st.metric("💰 Cap. Boursière", f"${market_cap/1e9:.1f}B")
            else:
                st.metric("💰 Cap. Boursière", "N/A")
        with col4:
            employees = info.get('fullTimeEmployees', 'N/A')
            st.metric("👥 Employés", f"{employees:,}" if isinstance(employees, int) else employees)
        
        st.markdown("---")
        
        # Calcul des indicateurs
        with st.spinner('📊 Calcul des indicateurs techniques...'):
            df = calculate_indicators(data)
        
        # Affichage des métriques clés
        st.header("📊 Métriques Actuelles")
        
        latest = df.iloc[-1]
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            price_change = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            st.metric("💲 Prix", f"${latest['Close']:.2f}", f"{price_change:+.2f}%")
        
        with col2:
            rsi_color = "🔴" if latest['RSI'] > 70 else "🟢" if latest['RSI'] < 30 else "🟡"
            st.metric(f"{rsi_color} RSI", f"{latest['RSI']:.1f}")
        
        with col3:
            macd_status = "📈" if latest['MACD'] > 0 else "📉"
            st.metric(f"{macd_status} MACD", f"{latest['MACD']:.4f}")
        
        with col4:
            st.metric("💨 Volatilité", f"{latest['Volatility']:.1f}%")
        
        with col5:
            total_return = ((latest['Close'] / df['Close'].iloc[0] - 1) * 100)
            st.metric(f"📈 Rdt Total ({period_label})", f"{total_return:+.1f}%")
        
        st.markdown("---")
        
        # Graphique principal
        st.header("📈 Analyse Technique Complète")
        with st.spinner('📊 Génération du graphique principal...'):
            main_chart = create_main_chart(df, ticker)
            st.plotly_chart(main_chart, use_container_width=True)
        
        st.markdown("---")
        
        # Graphique de volatilité
        st.header("💨 Analyse de Volatilité et Rendements")
        with st.spinner('📊 Génération de l\'analyse de volatilité...'):
            vol_chart = create_volatility_chart(df, ticker)
            st.plotly_chart(vol_chart, use_container_width=True)
        
        # Statistiques de rendements
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Rendement Moyen Journalier", f"{df['Daily_Return'].mean()*100:.3f}%")
        with col2:
            st.metric("📊 Écart-type", f"{df['Daily_Return'].std()*100:.3f}%")
        with col3:
            sharpe = (df['Daily_Return'].mean() / df['Daily_Return'].std()) * np.sqrt(252)
            st.metric("⚡ Ratio de Sharpe", f"{sharpe:.3f}")
        
        st.markdown("---")
        
        # Graphique Support/Résistance
        st.header("🎯 Niveaux Support/Résistance")
        with st.spinner('📊 Génération de l\'analyse support/résistance...'):
            sr_chart = create_support_resistance_chart(df, ticker)
            st.plotly_chart(sr_chart, use_container_width=True)
        
        # Niveaux techniques
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔻 Support", f"${latest['Support']:.2f}")
        with col2:
            st.metric("🔺 Résistance", f"${latest['Resistance']:.2f}")
        with col3:
            st.metric("💰 VWAP", f"${latest['VWAP']:.2f}")
        with col4:
            support_dist = ((latest['Close'] - latest['Support']) / latest['Close']) * 100
            st.metric("📏 Distance Support", f"{support_dist:.1f}%")
        
        st.markdown("---")
        
        # Modélisation prédictive
        st.header("🤖 Analyse Prédictive (Machine Learning)")
        
        with st.spinner('🤖 Entraînement du modèle Random Forest...'):
            rf_model, feature_columns, test_data, rf_pred, rf_r2, rf_mse, feature_importance = create_prediction_model(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 R² Score", f"{rf_r2:.4f}")
        with col2:
            st.metric("📉 MSE", f"{rf_mse:.6f}")
        
        # Feature importance
        st.subheader("🔍 Top 10 Features Importantes")
        top_features = feature_importance.head(10)
        
        fig_importance = go.Figure()
        fig_importance.add_trace(
            go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker_color='#45aaf2'
            )
        )
        fig_importance.update_layout(
            template='plotly_dark',
            height=400,
            xaxis_title="Importance",
            yaxis_title="Feature"
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Prédiction pour le prochain jour
        st.subheader("🔮 Prédiction pour le Prochain Jour de Trading")
        
        # Préparer les dernières features
        features_df = df.copy()
        lookback_days = 5
        
        for i in range(1, lookback_days + 1):
            features_df[f'Close_lag_{i}'] = features_df['Close'].shift(i)
            features_df[f'Volume_lag_{i}'] = features_df['Volume'].shift(i)
            features_df[f'Return_lag_{i}'] = features_df['Daily_Return'].shift(i)
        
        features_df['Price_MA20_ratio'] = features_df['Close'] / features_df['MA_20']
        features_df['Price_MA50_ratio'] = features_df['Close'] / features_df['MA_50']
        features_df['RSI_level'] = features_df['RSI']
        features_df['MACD_signal'] = (features_df['MACD'] > features_df['MACD_Signal']).astype(int)
        features_df['BB_position'] = (features_df['Close'] - features_df['BB_Lower']) / (features_df['BB_Upper'] - features_df['BB_Lower'])
        features_df['Vol_ratio'] = features_df['Volatility'] / features_df['Volatility'].mean()
        
        latest_features = features_df[feature_columns].iloc[-1:].dropna(axis=1)
        
        if not latest_features.empty:
            common_features = [col for col in feature_columns if col in latest_features.columns]
            latest_X = latest_features[common_features]
            
            if len(latest_X.columns) > 0:
                next_day_pred = rf_model.predict(latest_X.values.reshape(1, -1))[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📈 Rendement Prédit", f"{next_day_pred*100:.2f}%")
                
                with col2:
                    direction = "HAUSSIÈRE 📈" if next_day_pred > 0 else "BAISSIÈRE 📉"
                    st.metric("🎯 Direction", direction)
                
                with col3:
                    current_price = df['Close'].iloc[-1]
                    target_price = current_price * (1 + next_day_pred)
                    st.metric("💰 Prix Cible Estimé", f"${target_price:.2f}")
                
                # Niveau de confiance
                confidence_level = "FORTE ⭐⭐⭐" if abs(next_day_pred) > 0.02 else "MODÉRÉE ⭐⭐" if abs(next_day_pred) > 0.01 else "FAIBLE ⭐"
                st.info(f"🎯 Niveau de confiance: {confidence_level}")
        
        st.markdown("---")
        
        # Rapport de synthèse
        st.header("🎯 Rapport de Synthèse")
        
        # Analyse de tendance
        current_price = df['Close'].iloc[-1]
        ma20 = df['MA_20'].iloc[-1]
        ma50 = df['MA_50'].iloc[-1]
        ma200 = df['MA_200'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        current_macd = df['MACD'].iloc[-1]
        current_vol = df['Volatility'].iloc[-1]
        avg_vol = df['Volatility'].mean()
        
        # Signaux haussiers
        bullish_signals = sum([
            current_price > ma20,
            current_price > ma50,
            current_price > ma200,
            current_macd > 0,
            30 <= current_rsi <= 70
        ])
        
        # Tendance globale
        if bullish_signals >= 4:
            trend = "🟢 HAUSSIÈRE FORTE"
            trend_color = "success"
        elif bullish_signals >= 3:
            trend = "🟡 HAUSSIÈRE MODÉRÉE"
            trend_color = "warning"
        elif bullish_signals >= 2:
            trend = "🟠 NEUTRE"
            trend_color = "info"
        else:
            trend = "🔴 BAISSIÈRE"
            trend_color = "error"
        
        st.subheader(f"Tendance Globale: {trend}")
        
        # Analyse détaillée
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Analyse Tendancielle:**")
            st.write(f"- Prix vs MA20: {'✅' if current_price > ma20 else '❌'} ${ma20:.2f}")
            st.write(f"- Prix vs MA50: {'✅' if current_price > ma50 else '❌'} ${ma50:.2f}")
            st.write(f"- Prix vs MA200: {'✅' if current_price > ma200 else '❌'} ${ma200:.2f}")
            
            st.markdown("**⚡ Indicateurs:**")
            rsi_status = "SURACHETÉ ⚠️" if current_rsi > 70 else "SURVENDU ✅" if current_rsi < 30 else "NEUTRE"
            st.write(f"- RSI ({current_rsi:.1f}): {rsi_status}")
            macd_status = "HAUSSIER ✅" if current_macd > 0 else "BAISSIER ⚠️"
            st.write(f"- MACD: {macd_status}")
            vol_status = "ÉLEVÉE ⚠️" if current_vol > avg_vol * 1.2 else "NORMALE ✅"
            st.write(f"- Volatilité: {vol_status}")
        
        with col2:
            st.markdown("**💡 Recommandations:**")
            
            if bullish_signals >= 4:
                st.success("✅ ACHAT RECOMMANDÉ")
                st.write("- Tendance haussière confirmée")
                st.write("- Momentum positif")
            elif bullish_signals >= 3:
                st.warning("🟡 ACHAT AVEC PRUDENCE")
                st.write("- Tendance globalement positive")
                st.write("- Surveiller les résistances")
            elif bullish_signals >= 2:
                st.info("⏸️ ATTENTE")
                st.write("- Signaux mixtes")
                st.write("- Attendre une direction claire")
            else:
                st.error("❌ ÉVITER/VENDRE")
                st.write("- Tendance baissière dominante")
                st.write("- Risque élevé")
            
            st.markdown("**🎯 Niveaux Stratégiques:**")
            support = df['Support'].iloc[-1]
            resistance = df['Resistance'].iloc[-1]
            st.write(f"- 🟢 Zone d'achat: ${support:.2f} - ${support*1.02:.2f}")
            st.write(f"- 🔴 Stop Loss: ${support*0.98:.2f}")
            st.write(f"- 🎯 Objectif 1: ${resistance*.98:.2f}")
            st.write(f"- 🎯 Objectif 2: ${resistance*1.02:.2f}")
        
        # Performance historique
        st.markdown("---")
        st.subheader("📊 Performance Historique")
        
        col1, col2, col3 = st.columns(3)
        
        total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
        annualized_return = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (252/len(df)) - 1) * 100
        max_dd = ((df['Close'].cummax() - df['Close']) / df['Close'].cummax()).max() * 100
        
        with col1:
            st.metric("📈 Rendement Total", f"{total_return:+.1f}%")
        with col2:
            st.metric("📅 Rendement Annualisé", f"{annualized_return:+.1f}%")
        with col3:
            st.metric("📉 Drawdown Maximum", f"-{max_dd:.1f}%")
        
        # Avertissement
        st.markdown("---")
        st.warning("⚠️ **Avertissement:** Cette analyse est basée sur des données historiques et ne constitue pas un conseil en investissement. Toujours faire ses propres recherches avant d'investir.")
        
    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
        st.info("💡 Vérifiez que le ticker est valide et réessayez.")

else:
    # Page d'accueil
    st.info("👈 Entrez un ticker dans la barre latérale et cliquez sur 'Lancer l'Analyse' pour commencer!")
    
    st.markdown("## 📚 Guide d'utilisation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Fonctionnalités
        - **Analyse technique complète** avec tous les indicateurs
        - **Visualisations interactives** (Prix, MACD, RSI, Volume)
        - **Analyse de volatilité** et rendements
        - **Support/Résistance** automatiques
        - **Machine Learning** pour prédictions
        - **Recommandations** d'achat/vente
        """)
    
    with col2:
        st.markdown("""
        ### 💡 Exemples de tickers
        - **AAPL** - Apple Inc.
        - **TSLA** - Tesla
        - **NVDA** - Nvidia
        - **GOOGL** - Google
        - **MSFT** - Microsoft
        - **MU** - Micron Technology
        """)
    
    st.markdown("""
    ### 📊 Indicateurs Techniques Inclus
    - **Moyennes Mobiles:** MA20, MA50, MA200
    - **MACD:** Moving Average Convergence Divergence
    - **RSI:** Relative Strength Index (14 jours)
    - **Bandes de Bollinger:** Volatilité et niveaux de prix
    - **VWAP:** Volume Weighted Average Price
    - **Support/Résistance:** Niveaux techniques clés
    
    ### 🤖 Machine Learning
    - Modèle **Random Forest** avec 100 arbres
    - Prédiction des rendements du lendemain
    - Analyse d'importance des features
    - Évaluation avec R² et MSE
    """)
