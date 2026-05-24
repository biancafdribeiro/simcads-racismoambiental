# ============================================================
#  RACISMO AMBIENTAL NO BRASIL – Análise de Machine Learning
#  Variável resposta: tx_internacao_100k
#  Preditores: tx_esgoto_pct, prop_negra_parda_pct, regiao_cod,
#              capital_cod, porte_cod
#  Métodos: Regressão | Árvore de Decisão | Clustering (K-Means)
# ============================================================

# ── 0. INSTALAÇÃO (só necessário no Colab) ───────────────────
# !pip install pandas numpy scikit-learn matplotlib seaborn openpyxl -q

# ── 1. IMPORTS ───────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

# ── 2. CARREGAR DADOS ────────────────────────────────────────
# Se estiver no Colab, faça upload do arquivo e ajuste o caminho:
# from google.colab import files
# files.upload()

ARQUIVO = 'base_preparada_racismo_ambiental_261mun.xlsx'

df = pd.read_excel(ARQUIVO, sheet_name='Base_Preparada')
print(f"Base carregada: {df.shape[0]} municípios × {df.shape[1]} variáveis")

# ── 3. SELEÇÃO DE VARIÁVEIS ──────────────────────────────────
Y_COL  = 'tx_internacao_100k'
X_COLS = ['tx_esgoto_pct', 'prop_negra_parda_pct', 'regiao_cod',
          'capital_cod', 'porte_cod']

# Remover municípios com Y ausente (3 com 0 internações)
df_model = df[X_COLS + [Y_COL, 'nome_municipio', 'sigla_uf', 'regiao']].dropna().copy()
print(f"Municípios usados nos modelos: {len(df_model)}")

X = df_model[X_COLS]
y = df_model[Y_COL]

# ============================================================
#  BLOCO A – REGRESSÃO LINEAR MÚLTIPLA
# ============================================================
print("\n" + "="*60)
print("BLOCO A – REGRESSÃO LINEAR MÚLTIPLA")
print("="*60)

reg = LinearRegression()
reg.fit(X, y)
y_pred_reg = reg.predict(X)

# Métricas
r2   = r2_score(y, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y, y_pred_reg))
mae  = mean_absolute_error(y, y_pred_reg)

# Validação cruzada (5-fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(reg, X, y, cv=kf, scoring='r2')

print(f"\n📊 Métricas de desempenho:")
print(f"   R²              : {r2:.4f}")
print(f"   R² médio (5-CV) : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"   RMSE            : {rmse:.2f}")
print(f"   MAE             : {mae:.2f}")

print(f"\n📌 Coeficientes da regressão:")
coef_df = pd.DataFrame({
    'Variável': X_COLS,
    'Coeficiente': reg.coef_,
    'Interpretação': [
        'A cada +1pp de esgoto, a taxa de internação muda em...',
        'A cada +1pp de pop. negra/parda, a taxa muda em...',
        'Efeito médio de estar em uma região mais ao sul',
        'Diferença entre capital e não-capital',
        'Efeito de ser um município maior'
    ]
}).sort_values('Coeficiente', key=abs, ascending=False)
coef_df['Coeficiente'] = coef_df['Coeficiente'].round(4)
print(coef_df.to_string(index=False))
print(f"\n   Intercepto: {reg.intercept_:.4f}")

# ── Gráfico A1: Coeficientes ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('REGRESSÃO LINEAR MÚLTIPLA', fontsize=14, fontweight='bold', y=1.02)

cores = ['#d62728' if c < 0 else '#1f77b4' for c in coef_df['Coeficiente']]
axes[0].barh(coef_df['Variável'], coef_df['Coeficiente'], color=cores, edgecolor='white')
axes[0].axvline(0, color='black', linewidth=0.8, linestyle='--')
axes[0].set_title('Coeficientes (magnitude = importância)', fontsize=11)
axes[0].set_xlabel('Valor do coeficiente')
for i, v in enumerate(coef_df['Coeficiente']):
    axes[0].text(v + (0.3 if v >= 0 else -0.3), i, f'{v:.2f}',
                 va='center', ha='left' if v >= 0 else 'right', fontsize=9)

# Gráfico A2: Previsto vs Real
axes[1].scatter(y, y_pred_reg, alpha=0.6, color='#2196F3', edgecolors='white', linewidth=0.5)
lim = [min(y.min(), y_pred_reg.min()) - 5, max(y.max(), y_pred_reg.max()) + 5]
axes[1].plot(lim, lim, 'r--', linewidth=1.5, label='Linha perfeita')
axes[1].set_xlabel('Taxa real (por 100k hab)')
axes[1].set_ylabel('Taxa prevista')
axes[1].set_title(f'Previsto vs Real  |  R² = {r2:.3f}', fontsize=11)
axes[1].legend()
axes[1].set_xlim(lim); axes[1].set_ylim(lim)

plt.tight_layout()
plt.savefig('regressao_resultados.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico salvo: regressao_resultados.png")

# ── Interpretação automática ──────────────────────────────────
print("\n💬 Interpretação dos coeficientes:")
for _, row in coef_df.iterrows():
    direcao = "aumenta" if row['Coeficiente'] > 0 else "reduz"
    print(f"   → {row['Variável']:25s}: cada +1 unidade {direcao} a taxa em {abs(row['Coeficiente']):.2f} internações/100k")

# ============================================================
#  BLOCO B – ÁRVORE DE DECISÃO
# ============================================================
print("\n" + "="*60)
print("BLOCO B – ÁRVORE DE DECISÃO")
print("="*60)

# Testar diferentes profundidades
print("\n🔍 Seleção de profundidade (R² em validação cruzada):")
resultados_depth = []
for d in range(2, 8):
    tree = DecisionTreeRegressor(max_depth=d, random_state=42)
    scores = cross_val_score(tree, X, y, cv=kf, scoring='r2')
    resultados_depth.append({'max_depth': d, 'R2_mean': scores.mean(), 'R2_std': scores.std()})
    print(f"   depth={d}: R²={scores.mean():.4f} ± {scores.std():.4f}")

# Escolher melhor profundidade
best = max(resultados_depth, key=lambda x: x['R2_mean'])
BEST_DEPTH = best['max_depth']
print(f"\n   ✅ Melhor profundidade: {BEST_DEPTH} (R²={best['R2_mean']:.4f})")

tree = DecisionTreeRegressor(max_depth=BEST_DEPTH, random_state=42)
tree.fit(X, y)
y_pred_tree = tree.predict(X)

r2_tree   = r2_score(y, y_pred_tree)
rmse_tree = np.sqrt(mean_squared_error(y, y_pred_tree))
mae_tree  = mean_absolute_error(y, y_pred_tree)
cv_tree   = cross_val_score(tree, X, y, cv=kf, scoring='r2')

print(f"\n📊 Métricas (depth={BEST_DEPTH}):")
print(f"   R²              : {r2_tree:.4f}")
print(f"   R² médio (5-CV) : {cv_tree.mean():.4f} ± {cv_tree.std():.4f}")
print(f"   RMSE            : {rmse_tree:.2f}")
print(f"   MAE             : {mae_tree:.2f}")

# Importância das variáveis
imp_df = pd.DataFrame({
    'Variável': X_COLS,
    'Importância': tree.feature_importances_
}).sort_values('Importância', ascending=False)
print(f"\n📌 Importância das variáveis (Gini):")
print(imp_df.round(4).to_string(index=False))

# ── Gráfico B1: Árvore visual ─────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 6))
plot_tree(tree, feature_names=X_COLS, filled=True, rounded=True,
          fontsize=9, ax=ax, impurity=False, precision=1)
ax.set_title(f'Árvore de Decisão (profundidade={BEST_DEPTH}) | Racismo Ambiental',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('arvore_decisao.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico salvo: arvore_decisao.png")

# ── Gráfico B2: Importância das variáveis ────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
cores_imp = sns.color_palette('Blues_r', len(imp_df))
ax.barh(imp_df['Variável'], imp_df['Importância'], color=cores_imp, edgecolor='white')
ax.set_title('Importância das Variáveis – Árvore de Decisão', fontweight='bold')
ax.set_xlabel('Importância relativa (Gini)')
ax.axvline(0, color='black', linewidth=0.5)
for i, v in enumerate(imp_df['Importância']):
    ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('arvore_importancia.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico salvo: arvore_importancia.png")

# Regras textuais
print("\n📋 Regras da árvore (texto):")
print(export_text(tree, feature_names=X_COLS))

# ============================================================
#  BLOCO C – CLUSTERING (K-MEANS)
# ============================================================
print("\n" + "="*60)
print("BLOCO C – CLUSTERING K-MEANS")
print("="*60)

# Variáveis para clustering (as mesmas + tx_internacao como contexto)
CLUSTER_COLS = ['tx_esgoto_pct', 'prop_negra_parda_pct', 'tx_internacao_100k', 'porte_cod']
df_cluster = df_model[CLUSTER_COLS].copy()

# Normalizar (obrigatório para K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

# ── Método do cotovelo para escolher K ───────────────────────
print("\n🔍 Método do cotovelo (WCSS por K):")
wcss = []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
    print(f"   K={k}: WCSS={km.inertia_:.1f}")

# Escolha automática por maior queda relativa (cotovelo)
diffs = [wcss[i-1] - wcss[i] for i in range(1, len(wcss))]
BEST_K = list(K_range)[diffs.index(max(diffs))]
print(f"\n   ✅ K sugerido pelo cotovelo: {BEST_K}")

# Rodar K-Means com melhor K
km_final = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
df_model['cluster'] = km_final.fit_predict(X_scaled)

# ── Perfil de cada cluster ────────────────────────────────────
perfil = df_model.groupby('cluster')[CLUSTER_COLS + ['cluster']].agg(
    esgoto_medio=('tx_esgoto_pct', 'mean'),
    prop_negra_media=('prop_negra_parda_pct', 'mean'),
    tx_internacao_media=('tx_internacao_100k', 'mean'),
    porte_medio=('porte_cod', 'mean'),
    n_municipios=('cluster', 'count')
).round(2)

print(f"\n📊 Perfil dos {BEST_K} clusters:")
print(perfil.to_string())

# Nomear clusters automaticamente com base no perfil
def nomear_cluster(row):
    alto_risco = row['esgoto_medio'] < 40 and row['prop_negra_media'] > 65
    baixo_risco = row['esgoto_medio'] > 70 and row['prop_negra_media'] < 55
    if alto_risco:
        return 'Alto risco (saneamento precário + alta pop. negra)'
    elif baixo_risco:
        return 'Baixo risco (saneamento adequado + menor prop. negra)'
    elif row['tx_internacao_media'] > 80:
        return 'Alto adoecimento'
    elif row['esgoto_medio'] < 50:
        return 'Saneamento intermediário vulnerável'
    else:
        return 'Perfil intermediário'

perfil['Perfil'] = perfil.apply(nomear_cluster, axis=1)
df_model['cluster_nome'] = df_model['cluster'].map(perfil['Perfil'])

print("\n📌 Nomes atribuídos aos clusters:")
for c, nome in perfil['Perfil'].items():
    n = perfil.loc[c, 'n_municipios']
    print(f"   Cluster {c} ({n} municípios): {nome}")

# ── Gráfico C1: Cotovelo ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CLUSTERING K-MEANS', fontsize=14, fontweight='bold')

axes[0].plot(list(K_range), wcss, 'bo-', linewidth=2, markersize=7)
axes[0].axvline(BEST_K, color='red', linestyle='--', label=f'K={BEST_K} (cotovelo)')
axes[0].set_xlabel('Número de clusters (K)')
axes[0].set_ylabel('WCSS (inércia)')
axes[0].set_title('Método do Cotovelo')
axes[0].legend()
axes[0].grid(alpha=0.3)

# ── Gráfico C2: Scatter esgoto × prop_negra colorido por cluster
palette = sns.color_palette('Set1', BEST_K)
for c in range(BEST_K):
    mask = df_model['cluster'] == c
    axes[1].scatter(
        df_model.loc[mask, 'tx_esgoto_pct'],
        df_model.loc[mask, 'prop_negra_parda_pct'],
        c=[palette[c]], label=f'Cluster {c}',
        alpha=0.75, edgecolors='white', linewidth=0.4, s=60
    )
axes[1].set_xlabel('Cobertura de esgoto (%)')
axes[1].set_ylabel('Proporção pop. negra/parda (%)')
axes[1].set_title('Clusters: Esgoto × Raça')
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)
# linhas de referência
axes[1].axvline(40, color='gray', linestyle=':', alpha=0.6, label='Limiar crítico 40%')
axes[1].axhline(70, color='gray', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('clustering_kmeans.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico salvo: clustering_kmeans.png")

# ── Gráfico C3: Heatmap do perfil de clusters ─────────────────
perfil_heat = perfil[['esgoto_medio','prop_negra_media','tx_internacao_media','porte_medio']]
perfil_norm = (perfil_heat - perfil_heat.min()) / (perfil_heat.max() - perfil_heat.min())

fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(perfil_norm.T, annot=perfil_heat.T.round(1), fmt='.1f',
            cmap='RdYlGn_r', ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Valor normalizado (0=menor, 1=maior)'})
ax.set_title('Perfil dos Clusters (valores originais anotados)', fontweight='bold')
ax.set_xticklabels([f'Cluster {i}\n({perfil.loc[i,"n_municipios"]} mun.)' for i in perfil.index])
ax.set_yticklabels(['Esgoto (%)','Pop. Negra (%)','Taxa internação','Porte'], rotation=0)
plt.tight_layout()
plt.savefig('clusters_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico salvo: clusters_heatmap.png")

# ============================================================
#  BLOCO D – COMPARATIVO FINAL DOS MODELOS
# ============================================================
print("\n" + "="*60)
print("BLOCO D – COMPARATIVO FINAL")
print("="*60)

comparativo = pd.DataFrame({
    'Modelo': ['Regressão Linear', f'Árvore de Decisão (depth={BEST_DEPTH})'],
    'R²':     [round(r2, 4),       round(r2_tree, 4)],
    'R² CV':  [round(cv_r2.mean(), 4), round(cv_tree.mean(), 4)],
    'RMSE':   [round(rmse, 2),     round(rmse_tree, 2)],
    'MAE':    [round(mae, 2),      round(mae_tree, 2)],
})
print(comparativo.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(comparativo))
w = 0.35
bars1 = ax.bar(x - w/2, comparativo['R²'],    w, label='R² treino', color='#1f77b4')
bars2 = ax.bar(x + w/2, comparativo['R² CV'], w, label='R² validação cruzada', color='#ff7f0e')
ax.set_xticks(x)
ax.set_xticklabels(comparativo['Modelo'])
ax.set_ylabel('R²')
ax.set_ylim(0, 1)
ax.set_title('Comparativo de Modelos: R² Treino vs Validação Cruzada', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('comparativo_modelos.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico salvo: comparativo_modelos.png")

# ── Exportar resultados para Excel ───────────────────────────
print("\n💾 Exportando resultados...")
with pd.ExcelWriter('resultados_ml.xlsx', engine='openpyxl') as writer:
    comparativo.to_excel(writer, sheet_name='Comparativo_Modelos', index=False)
    coef_df[['Variável','Coeficiente']].to_excel(writer, sheet_name='Regressao_Coeficientes', index=False)
    imp_df.to_excel(writer, sheet_name='Arvore_Importancia', index=False)
    perfil.reset_index().to_excel(writer, sheet_name='Clusters_Perfil', index=False)
    df_model[['nome_municipio','sigla_uf','regiao',
              'tx_esgoto_pct','prop_negra_parda_pct','tx_internacao_100k',
              'cluster','cluster_nome']].to_excel(writer, sheet_name='Municipios_por_Cluster', index=False)

print("✅ Resultados exportados: resultados_ml.xlsx")

print("\n" + "="*60)
print("ANÁLISE CONCLUÍDA")
print(f"  Municípios analisados : {len(df_model)}")
print(f"  Variável resposta     : {Y_COL}")
print(f"  Preditores            : {', '.join(X_COLS)}")
print(f"  Melhor modelo (R² CV) : {'Regressão' if cv_r2.mean() >= cv_tree.mean() else 'Árvore'}")
print(f"  Clusters identificados: {BEST_K}")
print("="*60)
