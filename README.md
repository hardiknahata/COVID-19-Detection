# COVID-19-Detection using Chest X-Ray Images and Deep Learning

The Dataset used for training this model was made public by [Dr. Joseph Paul Cohen](https://josephpcohen.com/w/), a Post-Doctoral Fellow at University of Montréal, through an open-source database of COVID-19 chest X-Ray and CT Images on [this](https://github.com/ieee8023/covid-chestxray-dataset) GitHub Repo.

Dr. Cohen's Repo contains COVID-19, MERS, SARS, and ARDS infected patients' X-Ray Scans. I have extracted the COVID-19 Positive Samples from his Repo.
For the Healthy Patient Samples, I extracted images from the Kaggle Chest X-Ray Images Dataset.
I have saved both the COVID-19 Positive and Healthy X-Ray Samples in the [dataset](https://github.com/hardiknahata/COVID-19-Detection/dataset) folder in this GitHub Repo. The size of the dataset is very limited. The community is still working to enhance the dataset.
<br><br>
Here are samples from our COVID-19 patient Chest X-Ray Images Dataset.
<br>

COVID-19 Positive Samples                    |  COVID-19 Negative Samples
:-------------------------:|:-------------------------:
![COVID +](https://i.imgur.com/n6ZiH91.jpg)  |  ![COVID -](https://i.imgur.com/VXbW58d.jpg)

<br>

The [Data Prepare](https://github.com/hardiknahata/COVID-19-Detection/blob/master/Data%20Prepare.py) file contains the code to rename and preprocess the images in the dataset folder. Execute this code this before heading for the training code.

The [Train_Covid19](https://github.com/hardiknahata/COVID-19-Detection/blob/master/Train_Covid-19.py) file contains well documented code to build and train the deep learning model from scratch.
<br><br>
Given below is the Model Training & Testing Performance.
<br>
Accuracy Graph             | Loss Graph
:-------------------------:|:-------------------------:
![Acc](https://i.imgur.com/2qDMJhw.jpg)  |  ![Loss](https://i.imgur.com/KuTx3kY.jpg)

From the above plots we can observe that the Model does not Overfit even though our dataset is having limited training data.

<br>

Below is the Classification Report of the Deep Learning Model.
<br>
Classficiation Report
:-------------------------:|
![Classficiation Report](https://i.imgur.com/z7QwwoY.jpg)

From the above results, we can see that the Model has obtained _**93% Accuracy**_ on our dataset based only on X-Ray Images and no other data.

The Model has also obtained _**100% sensitivity**_ and _**88% specificity**_ which implies:
* Patients that are infected with COVID-19 (True Positives), have been accurately identified by the model as “COVID-19 Positive” _**100%**_ of the time.

* Patients that are NOT infected with COVID-19 (True Negatives), have been accurately identified by the model as “COVID-19 Negative” _**88%**_ of the time.

**NOTE**:
We are able to accurately detect COVID-19 with 100% Accuracy which is amazing, however, our true negative rate is a not convincing enough, for instance, we don’t want to classify someone as “COVID-19 negative” when they are “COVID-19 positive”.
<br><br>
**DISCLAIMER**<br>
The above project does not intend to 'solve' the COVID-19 Virus detection problem, nor does it claim to be a certified/authorized medical solution. The results obtained are shared just for educational purposes.


