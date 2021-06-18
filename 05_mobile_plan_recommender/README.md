# Recommendation of new plans for users of Megaline depending on user behavior

![image](https://user-images.githubusercontent.com/56832126/122574820-5cee1500-d050-11eb-8f3e-327cd373da04.png)


We are data scientists working for mobile operator Megaline. Recently. Mgealine has found that many of their subscribers (users) use legacy plans. However, the carrier want to analyze behavior of each user, and depending on their usage of data, minutes and text messages, Megaline want to recommend one of its new plans: Smart or Ultra.

Megaline provided us with a data which shows the behavior of subscribers who has recently switched to new plans. Based on these data, we are expected to build a model, which detects behavior of subscribers who use legacy plans and recommend them one of the new plans. Since we’ve already performed the data preprocessing step, we can move straight to creating the model.

Data description

Every observation in the dataset contains monthly behavior information about one user. The information given is as follows:
- `сalls` — number of calls,
- `minutes` — total call duration in minutes,
- `essages` — number of text messages,
- `mb_used` — Internet traffic used in MB,
- `is_ultra` — plan for the current month (Ultra - 1, Smart - 0).

Requirements:
- Develop a model with the highest possible accuracy. In this project, the threshold for accuracy is 0.75. Check the accuracy using the test dataset.

Steps we are going to follow:
- Load data file and check it;
- Split dataset into train, validation, and test sets;
- Apply different models, check their effectiveness;
- Check quality of the selected model using test set;
- Sanity Check
