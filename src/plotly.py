# ------------------------------------------------------------------------------
# SECTION 7: PLOTLY INTERACTIVE DASHBOARD & HTML EXPORT
# ------------------------------------------------------------------------------
print("\n[7/7] Generating interactive Plotly visual suite...")

COLOR_MAP = {
    'Positive': '#2ca02c',  # Green
    'Negative': '#d62728',  # Red
    'Neutral':  '#1f77b4'   # Blue
}

# 1. Aspect Breakdown Stacked Bar Chart
absa_counts = df_absa.groupby(['aspect', 'roberta_label']).size().reset_index(name='count')
fig_absa = px.bar(
    absa_counts,
    x='aspect',
    y='count',
    color='roberta_label',
    title='<b>1. Aspect-Based Sentiment Breakdown</b>',
    labels={'count': 'Review Count', 'aspect': 'Business Aspect', 'roberta_label': 'Sentiment'},
    color_discrete_map=COLOR_MAP,
    barmode='stack',
    text_auto=True
)
fig_absa.update_layout(template='plotly_white', font=dict(size=12))

# 2. Overall Portfolio Sentiment Donut
sentiment_counts = df['roberta_label'].value_counts().reset_index()
sentiment_counts.columns = ['sentiment', 'count']
fig_donut = px.pie(
    sentiment_counts,
    values='count',
    names='sentiment',
    title='<b>2. Overall Sentiment Distribution (Context & Sarcasm Aware)</b>',
    hole=0.45,
    color='sentiment',
    color_discrete_map=COLOR_MAP
)
fig_donut.update_traces(textposition='inside', textinfo='percent+label')
fig_donut.update_layout(template='plotly_white')

# 3. Item-Level Horizontal Mention Bars
pos_items = [item for items in df[df['roberta_label'] == 'Positive']['extracted_entities'] for item in items]
neg_items = [item for items in df[df['roberta_label'] == 'Negative']['extracted_entities'] for item in items]

top_pos = pd.DataFrame(Counter(pos_items).most_common(5), columns=['Item', 'Count'])
top_neg = pd.DataFrame(Counter(neg_items).most_common(5), columns=['Item', 'Count'])

fig_items = make_subplots(
    rows=1, cols=2,
    subplot_titles=('<b>Top Complimented Items</b>', '<b>Top Complained Items</b>')
)

if not top_pos.empty:
    fig_items.add_trace(
        go.Bar(x=top_pos['Count'], y=top_pos['Item'], orientation='h', marker_color='#2ca02c', text=top_pos['Count'], textposition='auto'),
        row=1, col=1
    )
if not top_neg.empty:
    fig_items.add_trace(
        go.Bar(x=top_neg['Count'], y=top_neg['Item'], orientation='h', marker_color='#d62728', text=top_neg['Count'], textposition='auto'),
        row=1, col=2
    )

fig_items.update_layout(template='plotly_white', showlegend=False, title_text='<b>3. Item & Menu-Level Specific Mentions</b>', height=400)
fig_items.update_yaxes(autorange="reversed")

# Render Charts in Colab
fig_absa.show()
fig_donut.show()
fig_items.show()

# Export to standalone HTML report
html_report_path = 'restaurant_nlp_report.html'
with open(html_report_path, 'w') as f:
    f.write(fig_absa.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write(fig_donut.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write(fig_items.to_html(full_html=False, include_plotlyjs='cdn'))

print(f"\n--> Standalone HTML report generated successfully as '{html_report_path}'.")
try:
    files.download(html_report_path)
except Exception:
    pass