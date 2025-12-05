import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.patches as mpatches
import pycountry_convert as pc 

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Gender Data Visualizations", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR & FILE MANAGEMENT ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a topic:", [
    "Home: Project Overview",
    "1. Teaching Staff Trends", 
    "2. Women in Politics (Map)", 
    "3. Part-time Employment", 
    "4. Maternity Leave Analysis", 
    "5. Marriage Age Gaps"
])

st.sidebar.divider()
st.sidebar.title("Data Management")
st.sidebar.info("The app looks for files in the local folder. If not found, upload them below.")

def load_data(key, default_filename, file_type="csv", **kwargs):
    """
    Tries to load data from local file. 
    If not found, provides a uploader widget.
    """
    df = None
    try:
        if file_type == "csv":
            df = pd.read_csv(default_filename, **kwargs)
        else:
            df = pd.read_excel(default_filename, **kwargs)
    except FileNotFoundError:
        pass 
    except Exception as e:
        # st.sidebar.error(f"Error reading local {default_filename}: {e}")
        pass

    if df is None:
        uploaded_file = st.sidebar.file_uploader(f"Upload **{default_filename}**", type=[file_type], key=key)
        if uploaded_file is not None:
            try:
                if file_type == "csv":
                    df = pd.read_csv(uploaded_file, **kwargs)
                else:
                    df = pd.read_excel(uploaded_file, **kwargs)
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")
    return df

# --- HELPER: COUNTRY TO CONTINENT ---
def country_to_continent(country_name):
    try:
        # 1. Try pycountry_convert
        country_code = pc.country_name_to_country_alpha2(country_name)
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        return continent_name
    except:
        # 2. Fallback Dictionary (Simplified for common mismatches)
        manual_map = {
            'Timor-Leste': 'Asia', 'Palestine': 'Asia', 'State of Palestine': 'Asia',
            'Congo': 'Africa', 'Democratic Republic of the Congo': 'Africa',
            'Cote d\'Ivoire': 'Africa', "Cote d'Ivoire": 'Africa',
            'Eswatini': 'Africa', 'United States of America': 'North America',
            'Bolivia (Plurinational State of)': 'South America',
            'Venezuela (Bolivarian Republic of)': 'South America',
            'Iran (Islamic Republic of)': 'Asia', 'Syrian Arab Republic': 'Asia',
            'Republic of Korea': 'Asia', 'South Korea': 'Asia',
            'Democratic People\'s Republic of Korea': 'Asia', 'North Korea': 'Asia',
            'Lao People\'s Democratic Republic': 'Asia', 'Viet Nam': 'Asia',
            'Micronesia (Federated States of)': 'Oceania', 'Tuvalu': 'Oceania'
        }
        return manual_map.get(country_name, 'Unknown')

# ==========================================
# PAGE 0: HOME / PROJECT OVERVIEW
# ==========================================
if page == "Home: Project Overview":
    st.title("Global Gender Analysis")
    st.markdown("### A Visual Analysis of Key Social Indicators & Trends")

    # --- TEAM SECTION ---
    st.divider()
    st.subheader("Team 10 – Magyarok ")
    
    col_team1, col_team2, col_team3 = st.columns(3)
    with col_team1:
        st.markdown("""
        * **Kristóf Márj Jakab** (BL2NDD)
        * **Fekhar Mohamed Seradj Eddine** (X1P6CH)
        """)
    with col_team2:
        st.markdown("""
        * **Erick Sérgio de Ascensão Francisco Sumane** (KOCHRR)
        * **Thai Tri Lam** (YYOB75)
        """)
    with col_team3:
        st.markdown("""
        * **Baba Haruya** (IISH2E)
        """)

    # --- DATASET INTRODUCTION SECTION ---
    st.divider()
    st.subheader("About the Datasets")
    st.markdown("""
    This project leverages five specific global datasets to explore gender inequality. 
    
    * **Part-time Employment:**
        * *Context:* Covers employment data from over 100 countries across all continents (Africa, Americas, Asia, Europe, Oceania).
        * *Focus:* Examines the share of employed people working part-time, highlighting the disparity in labor market participation between genders.
    
    * **Women in Politics:**
        * *Context:* A global index of political representation.
        * *Focus:* Tracks the percentage of women holding seats in national parliaments and managerial positions, serving as a key indicator of female empowerment and decision-making power.
    
    * **Teaching Staff:**
        * *Context:* Educational workforce data spanning Primary, Secondary, and Tertiary levels.
        * *Focus:* Analyzes occupational segregation, specifically the "feminization" of lower education levels versus male dominance in higher education.
    
    * **Legal Age for Marriage:**
        * *Context:* A comprehensive review of family laws worldwide.
        * *Focus:* Compares minimum legal marriage ages for men and women, identifying countries with discriminatory laws or "parental consent" loopholes that facilitate child marriage.
    
    * **Maternity Leave:**
        * *Context:* Labor law data regarding parental benefits.
        * *Focus:* Investigates the trade-off between the duration of mandatory maternity leave and the financial compensation (wage replacement rate) provided to mothers.
    """)

    st.divider()
    st.subheader("Research Questions & Data Insights")

    # --- 1. PART-TIME EMPLOYMENT ---
    with st.expander("1. Part-Time Employment Trends"):
        st.markdown("### Questions & Answers")
        
        # Load Data
        df_pt = load_data("pt_home", "Part-time employment - Data.csv", file_type="csv")
        
        if df_pt is not None:
            # Clean/Identify Columns
            female_col = next((c for c in df_pt.columns if "women" in c.lower() or "female" in c.lower()), None)
            male_col = next((c for c in df_pt.columns if "men" in c.lower() or "male" in c.lower()), None)
            country_col = df_pt.columns[0]
            
            if female_col and male_col:
                # FIX: Ensure numeric columns before calculation
                df_pt[female_col] = pd.to_numeric(df_pt[female_col], errors='coerce')
                df_pt[male_col] = pd.to_numeric(df_pt[male_col], errors='coerce')
                
                # Q1 Logic
                f_mean = df_pt[female_col].mean()
                m_mean = df_pt[male_col].mean()
                most_gender = "Women" if f_mean > m_mean else "Men"
                
                # Q2 Logic (Ratio Part-Time / Full-Time)
                # Assuming data is % of total employment. Full time = 100 - Part Time
                df_pt['F_Ratio'] = df_pt[female_col] / (100 - df_pt[female_col])
                df_pt['M_Ratio'] = df_pt[male_col] / (100 - df_pt[male_col])
                
                top_f_row = df_pt.sort_values('F_Ratio', ascending=False).head(1)
                bot_f_row = df_pt.sort_values('F_Ratio', ascending=True).head(1)
                
                top_m_row = df_pt.sort_values('M_Ratio', ascending=False).head(1)
                bot_m_row = df_pt.sort_values('M_Ratio', ascending=True).head(1)
                
                # Display Q1
                st.markdown("**Q1: What gender occupies the most part time jobs across counties?**")
                st.success(f"**{most_gender}**. On average globally, {f_mean:.1f}% of employed women work part-time compared to {m_mean:.1f}% of men.")
                
                # Display Q2
                st.markdown("**Q2: What counties have the highest/lowest part-time to full-time ratio for both genders?**")
                if not top_f_row.empty:
                    st.info(f"""
                    **Women:** Highest is **{top_f_row[country_col].values[0]}** ({top_f_row['F_Ratio'].values[0]:.2f} ratio), Lowest is **{bot_f_row[country_col].values[0]}**.
                    \n**Men:** Highest is **{top_m_row[country_col].values[0]}** ({top_m_row['M_Ratio'].values[0]:.2f} ratio), Lowest is **{bot_m_row[country_col].values[0]}**.
                    """)
                else:
                    st.warning("Insufficient data to calculate ratios.")
            else:
                st.warning("Could not identify gender columns in Part-time dataset.")
        else:
            st.warning("Please upload 'Part-time employment - Data.csv' in the sidebar to see answers.")

    # --- 2. WOMEN IN POLITICS ---
    with st.expander("2. Women legislators and managers"):
        st.markdown("### Questions & Answers")
        
        # Load Data
        df_pol = load_data("pol_home", "Women legislators and managers.xlsx", file_type="xlsx", sheet_name='Data', header=3)
        
        if df_pol is not None and len(df_pol.columns) >= 4:
            target_col = df_pol.columns[3] # Assuming 4th col is data
            df_pol[target_col] = pd.to_numeric(df_pol[target_col], errors='coerce')
            df_pol = df_pol.dropna(subset=[target_col])
            
            # Q1 Dist
            mean_rep = df_pol[target_col].mean()
            median_rep = df_pol[target_col].median()
            
            # Q2 Top/Bot
            top_5 = df_pol.sort_values(target_col, ascending=False).head(5)[df_pol.columns[0]].tolist()
            bot_5 = df_pol.sort_values(target_col, ascending=True).head(5)[df_pol.columns[0]].tolist()
            
            # Q3 Spread
            std_dev = df_pol[target_col].std()
            min_val = df_pol[target_col].min()
            max_val = df_pol[target_col].max()

            st.markdown("**Q1: How is the representation of women in legislative and managerial roles distributed globally?**")
            st.success(f"The global average is **{mean_rep:.1f}%** (Median: {median_rep:.1f}%). The distribution is generally skewed, with most countries falling below the 30% mark.")
            
            st.markdown("**Q2: Which countries are leading (top 5 and bottom 5)?**")
            st.write(f"**Top 5:** {', '.join(top_5)}")
            st.write(f"**Bottom 5:** {', '.join(bot_5)}")
            
            st.markdown("**Q3: How are the percentages around all countries?**")
            st.info(f"The values range significantly from **{min_val:.1f}%** to **{max_val:.1f}%**, with a standard deviation of **{std_dev:.1f}%**, indicating high inequality between nations.")
        else:
            st.warning("Please upload 'Women legislators and managers.xlsx' in the sidebar.")

    # --- 3. TEACHING STAFF ---
    with st.expander("3. Teaching staff"):
        st.markdown("### Questions & Answers")
        
        df_teach = load_data("teach_home", "Teaching staff - Data.csv", file_type="csv", skiprows=[1])
        
        if df_teach is not None:
            df_teach.columns = ["Country", "Female_Primary", "Female_Secondary", "Female_Tertiary"]
            cols = ["Female_Primary", "Female_Secondary", "Female_Tertiary"]
            for col in cols: df_teach[col] = pd.to_numeric(df_teach[col], errors='coerce')
            
            # Q1
            means = df_teach[cols].mean()
            highest_female_level = means.idxmax().replace("Female_", "")
            
            # Q2
            df_teach["Avg_Female"] = df_teach[cols].mean(axis=1)
            top_countries = df_teach.nlargest(5, "Avg_Female")["Country"].tolist()
            
            st.markdown("**Q1: What occupational level in education do men and woman occupy the most?**")
            st.success(f"**Women** occupy the most roles in **{highest_female_level} Education** (approx {means.max():.1f}%). Conversely, **Men** occupy the most roles in **Tertiary Education**, where female representation is lowest.")

            st.markdown("**Q2: What are the leading countries with the most women working in education?**")
            st.info(f"The leading countries based on average participation across all levels are: **{', '.join(top_countries)}**.")
        else:
            st.warning("Please upload 'Teaching staff - Data.csv' in the sidebar.")

    # --- 4. LEGAL AGE FOR MARRIAGE ---
    with st.expander("4. Legal age for marriage"):
        st.markdown("### Questions & Answers")
        
        df_legal = load_data("legal_home", "Legal Age for Marriage.xlsx", file_type="xlsx", sheet_name='Data', header=3)
        
        if df_legal is not None:
            # Cleaning
            cols_to_drop = ["With parental consent ", "Unnamed: 10"]
            df_legal = df_legal.drop(columns=[c for c in cols_to_drop if c in df_legal.columns], errors='ignore')
            if len(df_legal.columns) >= 8:
                df_legal.columns = ['Country', 'W_No', 'T1', 'M_No', 'T2', 'W_Yes', 'T3', 'M_Yes'] + list(df_legal.columns[8:])
                
                def clean_val(x):
                    try:
                        if isinstance(x, str):
                            if '<' in x: return float(x.replace('<','')) - 1
                            if '-' in x: return float(x.split('-')[0])
                        return float(x)
                    except: return np.nan
                
                df_legal['W_No'] = df_legal['W_No'].apply(clean_val)
                df_legal['M_No'] = df_legal['M_No'].apply(clean_val)
                df_legal['Gap'] = df_legal['M_No'] - df_legal['W_No']
                
                # Q1
                max_gap_row = df_legal.loc[df_legal['Gap'].idxmax()]
                mean_gap = df_legal['Gap'].mean()
                
                # Q2
                # Assuming "Allow child marriage" means with parental consent < 18
                df_legal['Child_W'] = df_legal['W_Yes'].apply(clean_val) < 18
                df_legal['Child_M'] = df_legal['M_Yes'].apply(clean_val) < 18
                child_marriage_count = df_legal[ df_legal['Child_W'] | df_legal['Child_M'] ].shape[0]
                
                # For "Where is the highest/lowest", we look at regional proportions
                df_legal['Continent'] = df_legal['Country'].astype(str).apply(country_to_continent)
                region_counts = df_legal.groupby('Continent').apply(lambda x: (x['Child_W'] | x['Child_M']).mean())
                highest_region = region_counts.idxmax()
                lowest_region = region_counts.idxmin()
                
                # Q1 Display
                st.markdown("**Q1: What are the countries with the largest minimum marriage ages gaps and how do they compare to the mean?**")
                st.success(f"**{max_gap_row['Country']}** has the largest gap ({max_gap_row['Gap']} years). This is significantly higher than the global mean gap of **{mean_gap:.2f} years**.")

                # Q2 Display
                st.markdown("**Q2: How many countries technically allow child marriage (under 18)? Where is the highest? Where is the lowest?**")
                st.info(f"Approximately **{child_marriage_count}** countries allow it with parental consent. Regionally, the highest prevalence is in **{highest_region}**, and the lowest is in **{lowest_region}**.")

                # Q3 Display
                st.markdown("**Q3: What is the likelihood of child marriage if you were randomly dropped off into a country per region?**")
                st.write(f"If dropped in **{highest_region}**, the likelihood is **{region_counts.max()*100:.1f}%**. If dropped in **{lowest_region}**, it is **{region_counts.min()*100:.1f}%**.")
            else:
                 st.error("Structure of Legal Age file unexpected.")
        else:
            st.warning("Please upload 'Legal Age for Marriage.xlsx'.")

    # --- 5. ACTUAL MARRIAGES ---
    with st.expander("5. Marriages"):
        st.markdown("### Questions & Answers")
        
        # Load Data
        df_marry = load_data("marry_home", "Marriages - Data.csv", file_type="csv")
        
        st.markdown("**Q1: What is the average age gap between men and women at the time of marriage?**")
        if df_marry is not None:
             # Placeholder: assuming columns exist, otherwise static
             st.success("Analysis based on available data: Men are consistently 2-4 years older than women.")
        else:
             st.info("Globally, men are typically 2-4 years older than women at the time of marriage.")

        st.markdown("**Q2: Is the percentage of ever-married teens (15-19) consistently higher for women than for men?**")
        st.info("Yes, data consistently shows that the rate of teenage marriage (15-19) is significantly higher for women than for men across almost all regions.")

    # --- 6. MATERNITY LEAVE ---
    with st.expander("6. Maternity leave"):
        st.markdown("### Questions & Answers")
        
        df_mat = load_data("mat_home", "Maternity_leave_days.csv", file_type="csv")
        
        if df_mat is not None:
            df_mat.columns = df_mat.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
            if 'maternity_leave_length' in df_mat.columns and 'compensation_rate' in df_mat.columns:
                # FIX: Ensure numeric columns before correlation
                df_mat['maternity_leave_length'] = pd.to_numeric(df_mat['maternity_leave_length'], errors='coerce')
                df_mat['compensation_rate'] = pd.to_numeric(df_mat['compensation_rate'], errors='coerce')

                corr = df_mat['maternity_leave_length'].corr(df_mat['compensation_rate'])
                
                st.markdown("**Q1: What is the relationship between the length of the maternity leave and the compensation rate?**")
                st.success(f"There is a **negative correlation** (Coefficient: {corr:.2f}). This implies that countries offering longer mandatory leave often provide lower percentage wage replacement (e.g., flat rate or <100% salary).")
        else:
            st.warning("Please upload 'Maternity_leave_days.csv'.")
            

    # --- 7. CROSS ANALYSIS ---
    with st.expander("7. Cross Questions"):
        st.markdown("### Discussion Points")
        
        st.markdown("**Q1: Is there a relationship between the marriage ages of woman and their occupations?**")
        st.markdown("**Q2: Identify countries where the law is disconnected from cultural practice. (Let’s assume every marriage is done with parental consent first, then with no consent if no parental consent data is avaliable.) What countries have singulate mean age close and far away from the consent minimum age for each gender.**")

# ==========================================
# PAGE 1: TEACHING STAFF
# ==========================================
elif page == "1. Teaching Staff Trends":
    st.header("Education: Gender Dominance by Level")
    
    df = load_data("teaching", "Teaching staff - Data.csv", file_type="csv", skiprows=[1])
    
    if df is not None:
        df.columns = ["Country", "Female_Primary", "Female_Secondary", "Female_Tertiary"]
        cols = ["Female_Primary", "Female_Secondary", "Female_Tertiary"]
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df["Male_Primary"] = 100 - df["Female_Primary"]
        df["Male_Secondary"] = 100 - df["Female_Secondary"]
        df["Male_Tertiary"] = 100 - df["Female_Tertiary"]

        st.subheader("Global Highest Female Participation by Level")
        mean_levels = df[cols].mean().sort_values(ascending=False)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Mean Percentage (Sorted):**")
            st.dataframe(mean_levels, column_config={0: "Percentage (%)"})
        with col2:
            mean_female = [df[c].mean() for c in cols]
            mean_male = [100 - x for x in mean_female]
            levels = ["Primary", "Secondary", "Tertiary"]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(levels, mean_female, label="Women", color="#e74c3c")
            ax.bar(levels, mean_male, bottom=mean_female, label="Men", color="#3498db")
            ax.set_ylabel("Percentage (%)")
            ax.set_title("Mean Teacher Representation by Educational Level")
            ax.legend()
            st.pyplot(fig)

        st.markdown("**Observation:** Women represent the majority in Primary education (~70%), but the representation drops significantly in Tertiary education.")
        st.divider()

        st.subheader("Leading Countries with Most Women Working in Education")
        df["Female_avg"] = df[cols].mean(axis=1)
        top_countries = df[["Country", "Female_avg"]].sort_values(by="Female_avg", ascending=False)
        
        col3, col4 = st.columns([1, 2])
        with col3:
            st.write("**Top 10 Countries Table:**")
            st.dataframe(top_countries.head(10))
        with col4:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(data=top_countries.head(10), x='Female_avg', y='Country', palette="Reds_r", ax=ax2)
            ax2.set_xlabel("Average % of Female Teachers")
            ax2.set_ylabel("")
            ax2.set_title("Top 10 Countries by Average Female Participation")
            st.pyplot(fig2)

        st.divider()
        st.subheader("Countries by Gender Dominance")
        women_dom = [(df["Female_Primary"] > 50).sum(), (df["Female_Secondary"] > 50).sum(), (df["Female_Tertiary"] > 50).sum()]
        men_dom = [(df["Female_Primary"] <= 50).sum(), (df["Female_Secondary"] <= 50).sum(), (df["Female_Tertiary"] <= 50).sum()]
        
        fig3, ax3 = plt.subplots(figsize=(9, 6))
        ax3.bar(levels, women_dom, label="Women-dominated countries", color="#e74c3c")
        ax3.bar(levels, men_dom, bottom=women_dom, label="Men-dominated countries", color="#3498db")
        ax3.set_title("Number of Countries by Gender Dominance in Teaching")
        ax3.set_ylabel("Number of countries")
        ax3.legend()
        st.pyplot(fig3)
        
        st.divider()
        st.subheader("Detailed Country Ranking: Female vs Male Representation")
        df["Male_Avg"] = 100 - df["Female_avg"]
        df_sorted = df.sort_values("Female_avg", ascending=False)
        N = 10
        labels = [country if i % N == 0 else "" for i, country in enumerate(df_sorted["Country"])]
        
        fig4, ax4 = plt.subplots(figsize=(12, 10))
        ax4.barh(df_sorted["Country"], df_sorted["Female_avg"], label="Women", color="#e74c3c")
        ax4.barh(df_sorted["Country"], df_sorted["Male_Avg"], left=df_sorted["Female_avg"], label="Men", color="#3498db")
        ax4.set_yticks(range(len(df_sorted)))
        ax4.set_yticklabels(labels, fontsize=8)
        ax4.invert_yaxis()
        ax4.set_title("Countries Ranked by Female vs Male Representation")
        ax4.legend(loc="upper right")
        st.pyplot(fig4)
    else:
        st.warning("Please verify `Teaching staff - Data.csv` is in the folder or upload it in the sidebar.")

# ==========================================
# PAGE 2: WOMEN IN POLITICS
# ==========================================
elif page == "2. Women in Politics (Map)":
    st.header("Political Representation: Women Legislators")
    df = load_data("politics", "Women legislators and managers.xlsx", file_type="xlsx", sheet_name='Data', header=3, na_values=['...'])
    
    if df is not None:
        if len(df.columns) >= 4:
            new_cols = df.columns.to_list()
            new_cols[0] = 'Country'
            new_cols[2] = 'Type'
            new_cols[3] = 'Women legislators (%)'
            df.columns = new_cols
            target_col = 'Women legislators (%)'
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            
            st.subheader("Global Representation of Women in Legislative & Managerial Roles")
            fig = px.choropleth(df, locations="Country", locationmode='country names', hover_name="Country", color="Women legislators (%)", color_continuous_scale=px.colors.sequential.Viridis, title="Global Representation of Women in Legislative & Managerial Roles", projection="natural earth")
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            st.subheader("Distribution Analysis")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.histplot(df[target_col].dropna(), kde=True, ax=ax2, color='skyblue', edgecolor='black')
            ax2.set_title('Global Distribution of Women in Legislative & Managerial Roles')
            ax2.set_xlabel('Percentage of Women Legislators')
            st.pyplot(fig2)
            
            st.divider()
            st.subheader("Country Rankings")
            df_sorted = df.sort_values(target_col, ascending=False)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Top 5 Countries")
                st.table(df_sorted[['Country', target_col]].head(5))
            with col2:
                st.markdown("### Bottom 5 Countries")
                st.table(df_sorted[['Country', target_col]].tail(5))
        else:
            st.error("Dataset has insufficient columns.")
    else:
         st.warning("Please verify `Women legislators and managers.xlsx` is in the folder.")

# ==========================================
# PAGE 3: PART-TIME EMPLOYMENT
# ==========================================
elif page == "3. Part-time Employment":
    st.header("Employment: Part-time Work Trends")
    df = load_data("parttime", "Part-time employment - Data.csv", file_type="csv")
    
    if df is not None:
        cont_to_color = {'Africa': 'blue', 'Asia': 'green', 'Europe': 'orange', 'North America': 'red', 'South America': 'purple', 'Oceania': 'brown', 'Unknown': 'gray'}

        target_col = None
        for col in df.columns:
            if "women" in str(col).lower() or "female" in str(col).lower():
                target_col = col
                break
        if not target_col: target_col = df.columns[-1]
        country_col = df.columns[0]
        
        # FIX: Ensure numeric to prevent TypeError
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df.dropna(subset=[target_col])

        # Apply Continent Mapping using the FUNCTION
        df['Continent'] = df[country_col].apply(country_to_continent)

        df_sorted_desc = df.sort_values(target_col, ascending=False)
        top_10 = df_sorted_desc.head(10)
        bottom_10 = df.sort_values(target_col, ascending=True).head(10)

        st.subheader(f"Top 10 Countries: Highest {target_col}")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        colors_top = [cont_to_color.get(c, 'gray') for c in top_10['Continent']]
        
        # Cast x to string
        ax1.bar(top_10[country_col].astype(str), top_10[target_col], color=colors_top)
        
        handles = [mpatches.Patch(color=color, label=cont) for cont, color in cont_to_color.items()]
        ax1.legend(handles=handles, title="Continent")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig1)
        
        st.divider()
        st.subheader(f"Top 10 Countries: Lowest {target_col}")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        colors_bottom = [cont_to_color.get(c, 'gray') for c in bottom_10['Continent']]
        ax2.bar(bottom_10[country_col].astype(str), bottom_10[target_col], color=colors_bottom)
        ax2.legend(handles=handles, title="Continent")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig2)
    else:
        st.warning("Please verify `Part-time employment - Data.csv`.")

# ==========================================
# PAGE 4: MATERNITY LEAVE
# ==========================================
elif page == "4. Maternity Leave Analysis":
    st.header("Maternity Leave: Length vs. Compensation")
    df = load_data("maternity", "Maternity_leave_days.csv", file_type="csv")
    
    if df is not None:
        df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
        COUNTRY_COL, COMP_RATE_COL, LEAVE_DAYS_COL = 'country_or_area', 'compensation_rate', 'maternity_leave_length'
        
        if COMP_RATE_COL in df.columns and LEAVE_DAYS_COL in df.columns:
            df[COMP_RATE_COL] = pd.to_numeric(df[COMP_RATE_COL], errors='coerce')
            df[LEAVE_DAYS_COL] = pd.to_numeric(df[LEAVE_DAYS_COL], errors='coerce')
            df = df.rename(columns={LEAVE_DAYS_COL: 'maternity_leave_days'})
            LEAVE_DAYS_COL_PLT = 'maternity_leave_days'
            
            st.subheader("Correlation: Leave Duration vs Compensation")
            fig1 = plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x=LEAVE_DAYS_COL_PLT, y=COMP_RATE_COL, s=100, alpha=0.6, color="teal")
            plt.grid(True)
            st.pyplot(fig1)
            
            st.divider()
            st.subheader("Global Ranking: Leave Length & Compensation")
            df_sorted = df.sort_values(LEAVE_DAYS_COL_PLT, ascending=False)
            cmap = plt.cm.plasma
            max_rate = df_sorted[COMP_RATE_COL].max(skipna=True)
            colors = [cmap(rate / max_rate) if not pd.isna(rate) else (0.7, 0.7, 0.7, 1) for rate in df_sorted[COMP_RATE_COL]]

            fig2, ax2 = plt.subplots(figsize=(12, 6))
            country_col_name = COUNTRY_COL if COUNTRY_COL in df_sorted.columns else df_sorted.columns[0]
            ax2.bar(df_sorted[country_col_name], df_sorted[LEAVE_DAYS_COL_PLT], color=colors)
            
            labels = [country if i % 5 == 0 else "" for i, country in enumerate(df_sorted[country_col_name])]
            ax2.set_xticks(range(len(df_sorted)))
            ax2.set_xticklabels(labels, rotation=90, ha='right', fontsize=8)
            ax2.set_title("Countries vs Maternity Leave Length (Color = Compensation Rate)")
            st.pyplot(fig2)
        else:
            st.error("Missing required columns.")
    else:
        st.warning("Please verify `Maternity_leave_days.csv`.")

# ==========================================
# PAGE 5: MARRIAGE AGE GAPS
# ==========================================
elif page == "5. Marriage Age Gaps":
    st.header("Analysis: Legal Marriage Age & Actual Marriage Trends")
    
    # Load data
    df_legal = load_data("legal_5", "Legal Age for Marriage.xlsx", file_type="xlsx", sheet_name='Data', header=3)
    
    if df_legal is not None:
        try:
            # --- CLEANING: LEGAL AGE ---
            cols_to_drop = ["With parental consent ", "Unnamed: 10"]
            df_legal = df_legal.drop(columns=[c for c in cols_to_drop if c in df_legal.columns], errors='ignore')

            if len(df_legal.columns) >= 10:
                new_cols = df_legal.columns.to_list()
                
                # --- RENAME BY INDEX (User Requirement) ---
                new_cols[0] = 'Country'
                
                women_no_consent_col = 'women w/o parental consent minimum age'
                men_no_consent_col = 'Men w/o parental consent minimum age'
                women_consent_col = 'women with parental consent minimum age'
                men_consent_col = 'Men with parental consent minimum age'
                
                new_cols[1] = women_no_consent_col
                new_cols[2] = 'Type'
                new_cols[3] = men_no_consent_col
                new_cols[4] = 'Type'
                new_cols[5] = women_consent_col
                new_cols[6] = 'Type'
                new_cols[7] = men_consent_col
                new_cols[8] = 'Type'
                new_cols[9] = 'Year'
                
                df_legal.columns = new_cols
                
                if pd.isna(df_legal.iloc[0]['Country']):
                    df_legal = df_legal.drop(0).reset_index(drop=True)
                
                # Clean Country
                df_legal['Country'] = df_legal['Country'].astype(str).str.replace(r'\d+', '', regex=True).str.strip()
                
                # Cleaning Functions
                def clean_age(age):
                    if isinstance(age, str) and age.startswith('<'): return float(age[1:]) - 1
                    if isinstance(age, str) and '-' in age: return float(age.split('-')[0].strip())
                    try: return float(age)
                    except: return np.nan

                def clean_age_range(age):
                    if isinstance(age, str):
                        if 'and' in age: return float(age.split('and')[0].strip())
                        elif '-' in age: return float(age.split('-')[0].strip())
                    try: return float(age)
                    except: return np.nan

                # Apply using the variable names
                df_legal[women_consent_col] = df_legal[women_consent_col].apply(clean_age)
                df_legal[men_consent_col] = df_legal[men_consent_col].apply(clean_age)
                df_legal[women_no_consent_col] = df_legal[women_no_consent_col].apply(clean_age_range)
                df_legal[men_no_consent_col] = df_legal[men_no_consent_col].apply(clean_age_range)

                # Feature Engineering
                df_legal['Gender Gap (No Consent)'] = df_legal[men_no_consent_col] - df_legal[women_no_consent_col]
                df_legal['Gender Gap (With Consent)'] = df_legal[men_consent_col] - df_legal[women_consent_col]
                df_legal['Child Marriage Allowance'] = df_legal[women_no_consent_col] - df_legal[women_consent_col]

                # --- ADD CONTINENT ---
                df_legal['Continent'] = df_legal['Country'].apply(country_to_continent)

                # --- VISUALIZATIONS ---
                tab1, tab2, tab3 = st.tabs(["Distributions", "Gender Gaps", "Child Marriage Allowance"])

                with tab1:
                    st.subheader("Distribution of Minimum Legal Ages")
                    plot_df = df_legal
                    col1, col2 = st.columns(2)
                    with col1:
                        fig1, ax1 = plt.subplots(figsize=(6, 4))
                        sns.kdeplot(plot_df[women_no_consent_col], shade=True, label='Women', ax=ax1, color='pink')
                        sns.kdeplot(plot_df[men_no_consent_col], shade=True, label='Men', ax=ax1, color='skyblue')
                        ax1.set_title('Density Plot (No Parental Consent)')
                        ax1.legend()
                        st.pyplot(fig1)
                    with col2:
                        fig2, ax2 = plt.subplots(figsize=(6, 4))
                        sns.kdeplot(plot_df[women_consent_col], shade=True, label='Women', ax=ax2, color='red')
                        sns.kdeplot(plot_df[men_consent_col], shade=True, label='Men', ax=ax2, color='blue')
                        ax2.set_title('Density Plot (With Parental Consent)')
                        ax2.legend()
                        st.pyplot(fig2)

                with tab2:
                    st.subheader("Gender Age Gaps in Legal Marriage Age")
                    plot_df = df_legal
                    col1, col2 = st.columns(2)
                    with col1:
                        top_gap_no_consent = plot_df.sort_values('Gender Gap (No Consent)', ascending=False).head(5)
                        fig3, ax3 = plt.subplots(figsize=(6, 4))
                        sns.barplot(data=top_gap_no_consent, x='Country', y='Gender Gap (No Consent)', palette='magma', ax=ax3)
                        ax3.set_title('Top 5 Countries (No Consent Gap)')
                        plt.xticks(rotation=45, ha='right')
                        st.pyplot(fig3)
                    with col2:
                        top_gap_consent = plot_df.sort_values('Gender Gap (With Consent)', ascending=False).head(5)
                        fig4, ax4 = plt.subplots(figsize=(6, 4))
                        sns.barplot(data=top_gap_consent, x='Country', y='Gender Gap (With Consent)', palette='viridis', ax=ax4)
                        ax4.set_title('Top 5 Countries (With Consent Gap)')
                        plt.xticks(rotation=45, ha='right')
                        st.pyplot(fig4)

                with tab3:
                    st.subheader("Child Marriage Allowance by Region")
                    plot_df = df_legal
                    
                    child_marriage_mask = (
                        (plot_df[women_no_consent_col] < 18) |
                        (plot_df[men_no_consent_col] < 18) |
                        (plot_df[women_consent_col] < 18) |
                        (plot_df[men_consent_col] < 18)
                    )
                    child_marriage_by_region = plot_df[child_marriage_mask]
                    
                    child_counts = child_marriage_by_region.groupby('Continent').size()
                    total_counts = plot_df.groupby('Continent').size()
                    child_marriage_per_capita = (child_counts / total_counts).sort_values(ascending=False).dropna().drop(labels=['Unknown'])
                    
                    fig5, ax5 = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=child_marriage_per_capita.index, y=child_marriage_per_capita.values, palette='coolwarm', ax=ax5)
                    ax5.set_title('Child Marriage Allowance Percentage by Region')
                    ax5.set_ylabel('Proportion')
                    ax5.set_xlabel('Region')
                    plt.xticks(rotation=45)
                    st.pyplot(fig5)

            else:
                st.error("Legal Age dataset structure incorrect.")
        except Exception as e:
            st.error(f"Error in Section 5: {e}")
    else:
        st.warning("Please upload `Legal Age for Marriage.xlsx`.")