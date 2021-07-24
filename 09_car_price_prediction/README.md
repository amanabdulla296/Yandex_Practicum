# Optimal algorithm for price prediction
![image](https://user-images.githubusercontent.com/56832126/126864743-1f94efa9-3cbe-4227-a2a9-89952b0f29dd.png)


Rusty Bargain, a used car sales service, is developing an app to attract new customers. In that app, one can quickly find out the market value of his/her car. We have access to historical data: technical specifications, trim versions, and prices. Now, we need to build the model to determine the value. 

Rusty Bargain is interested in:

- the quality of the prediction;
- the time required for training

Based on above, we will consider different algorithms and tune their hyperparameters to find out optimal model, with highest accuracy with quick prediction.

Following steps will be carried out:
- 1. data loading and checking;
- 2. EDA
- 3. Training differnt models
- 4. Benchmarking: quality and speed


**Data description**

*Features*:

- `DateCrawled` — date profile was downloaded from the database
- `VehicleType` — vehicle body type
- `RegistrationYear` — vehicle registration year
- `Gearbox` — gearbox type
- `Power` — power (hp)
- `Model` — vehicle model
- `Mileage` — mileage (measured in km due to dataset's regional specifics)
- `RegistrationMonth` — vehicle registration month
- `FuelType` — fuel type
- `Brand` — vehicle brand
- `NotRepaired` — vehicle repaired or not
- `DateCreated` — date of profile creation
- `NumberOfPictures` — number of vehicle pictures
- `PostalCode` — postal code of profile owner (user)
- `LastSeen` — date of the last activity of the user

*Target*

- `Price` — price (Euro)
