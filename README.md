# BigGAN-Capstone-Project-Deliverables
This is the Hamoye Spring 2024 BigGAN Capstone data cleaning datasets, visualizations, target variables, modeling and deployment repository.

USING GENERATIVE AI TO FORECAST ELECTRICITY DEMAND AND SUPPLY GAPS: SUB-SAHARAN AFRICA VS. OTHER CONTINENT REGIONS
BigGAN Capstone Project Proposal, Spring ‘24 Cohort
Arthur Uzoma ⭐
Somi Fredrick 🌟
Abosede Omishade
Nina Garmash
Nishant Katiyar
Bharat Kumar Jhawar
Adeniyi Olaolu
Arnab Das
Sayantani Jana
Emmanuel Alafaa
Tolulope Oke
Ademo Vitalis
Abiodun Onifade
Oluchukwu Okorie
1. Introduction
Access to electricity is crucial for economic growth and social development. Yet in Sub-Saharan Africa (SSA), around 600 million people still need access to it, hindering the achievement of Sustainable Development Goal (SDG) 7, which targets universal access to modern energy by 2030.

To tackle this, the project aims to develop a generative AI model to predict electrification demand and supply gaps in SSA against other continent regions by understanding the factors influencing income group, region, access to alternative energy sources, balance values, and electricity rate and leveraging complex data patterns to provide actionable insights for improved planning and decision-making.

2. Existing work and References:
The previous cohort worked on developing a Global household predictive model specifically focusing on OECD countries. Their model predicts the Low, Medium, or High levels of electricity consumption by electricity access rate in the rural population(the most important feature), rural and urban population, year, and location. Another cohort worked on Predictive Modeling of Electricity Access in Africa, it analyzed historical trends relative to region and income across African countries and projected electricity access levels until 2030 (World Bank data). Two more groups focused on Predicting Future Electrification Needs using data from different datasets (Harvard Dataverse and World Bank).

Also, Forecasting Electrification Needs in Africa: A Multiscale Approach by Abigail B. McCoy, Jason L. Cohen, and Peter B. Lunt(2020). This research employs a multiscale modeling approach to forecast future African electrification needs. The authors combined macroeconomic trends with local-level data to create detailed electricity demand projections across different regions.

3. Problem Statement:
Despite significant efforts to improve electricity access, Sub-Saharan Africa (SSA) remains critically underserved. As of 2019, the number of people without access increased to 570 million, up from 556 million in 2010, highlighting a persistent and growing electrification gap. The issue of electricity in SSA is not just a demand issue; it is also a supply issue. Addressing this requires a comprehensive approach that includes predicting the demand/supply gap and comparing this metric against other regions of the world. This would provide insights into where the need is greatest and inform targeted interventions. Additionally, comparing these gaps to other continents or specific countries where electrification supply is not as critical can offer valuable lessons and strategies that could be adapted to the SSA context. There is a need for a generative AI model capable of accurately forecasting electrification demand, supply, and the resulting gap to guide effective and sustainable energy access interventions in SSA.

4. Objectives:
Comparative Analysis: Conduct a comparative analysis of the electrification gaps within SSA regions and between SSA and other continent regions or specific countries with higher electrification rates to extract valuable insights and strategies.
Guiding Interventions: Provide data-driven insights to guide targeted and effective interventions for improving sustainable energy access in SSA, considering both demand and supply gap factors.
Develop a Generative AI Model: create a robust Generative AI model to accurately forecast the electricity demand, supply, and resulting gap for SSA. This model will incorporate socio-economic, environmental, and technological factors to provide reliable predictions and actionable insights for improving electricity access and resource allocation in SSA.
Predict Demand/Supply Gap: Utilize machine learning techniques to forecast the electricity demand and supply gap for Sub-Saharan Africa. This approach will integrate socio-economic, environmental, and technological factors to predict areas with the highest needs and improve electricity access and resource planning.
Dynamic Adaptation: Ensure the AI model can dynamically adapt to new data inputs to continuously refine predictions and intervention strategies over time.
5. Methodology
5.1 Dataset:
Global Electrification Database by Harvard Dataverse - dataset.xhtml.
The World Bank data - Access to electricity (% of the population) and more - EG.ELC.ACCS.ZS.
World Energy Statistics and Balances IAE 50 - world-energy-statistics-and-balances.
TRACKING SDG 7 - The Energy Progress Report.
Electricity Production By Source (World), Kaggle.
5.2 Model Development
Data Preprocessing: Cleaning, normalizing, and integrating the datasets to create a comprehensive data repository.
Feature Selection: Identifying key features influencing electrification demand and supply using techniques like principal component analysis (PCA).
Model Training: Developing a generative AI model using techniques such as Variational Autoencoders (VAEs) or Generative Adversarial Networks (GANs) to generate predictive scenarios.
Validation and Testing: Using a portion of the data for validation and testing to ensure model accuracy. We will employ techniques like k-fold cross-validation to assess performance and prevent overfitting.
5.3 Model Deployment
Scenario Analysis: Generating multiple demand and supply scenarios to explore different future possibilities.
Interactive Dashboard: Creating a user-friendly interface to visualize predictions and facilitate scenario exploration.
Feedback Loop: Implementing a system to continuously update the model with new data for ongoing improvement.
6. Expected Outcomes
Accurate Forecasting: Develop a generative AI model to predict the electricity demand and supply gap for Sub-Saharan Africa (SSA). The model will consider economic indicators, population growth, weather patterns, and access to alternative energy sources for comprehensive forecasting.
Identification of Key Drivers: Generate insights into the primary drivers of electricity demand and supply across different regions. This will help to understand the unique challenges and opportunities in Sub-Saharan Africa compared to other continents.
Strategic Resource Allocation: Provide recommendations for optimizing resource allocation and infrastructure investments to address identified supply gaps efficiently. The insights will support targeted interventions to enhance electricity access where it is most needed.
Scenario Analysis Tools: Create an interactive tool to simulate various scenarios, allowing stakeholders to explore the impact of different policy decisions, economic conditions, and investment strategies on future electricity demand and supply.
Policy Recommendations: Offer data-driven policy recommendations to help governments, NGOs, and private sector stakeholders formulate effective strategies to improve electrification rates and manage supply gaps.
Enhanced Decision-Making: Equip energy planners and decision-makers with robust predictive insights to manage electricity supply more effectively, anticipate future demand, and respond to market dynamics, thereby reducing outages and improving energy reliability.
Comparative Analysis: Provide a comparative analysis of electrification challenges and solutions in Sub-Saharan Africa versus other continental regions, highlighting best practices and transferable strategies
7. Conclusion
In conclusion, implementing generative AI to forecast electricity demand and supply gaps in Sub-Saharan Africa offers a transformative approach to addressing persistent electrification challenges. By delivering precise predictions and comprehensive insights, this project will facilitate strategic planning, inform policy-making, and enable effective resource allocation. Additionally, the project will provide scenario analysis tools and enhance decision-making capabilities, offering a comparative perspective with other regions to transfer best practices and innovative solutions. Ultimately, leveraging generative AI in this context promises to drive sustainable and equitable electrification efforts, contributing to universal energy access and significantly improving socio-economic development across the region.

8. References:
 McCoy, A. B., Cohen, J. L., and Lunt, P. B., (2020). Forecasting Electrification Needs in Africa: A Multiscale Approach.
