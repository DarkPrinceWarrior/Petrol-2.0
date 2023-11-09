<h1 align="center"><Petrol prediction></h1>

<p align="center"><project-description></p>

## Screenshots

![Home Page](/static/images/preview.jpg "Home Page")

![](/screenshots/2.png)
 
  
## Work process
Программный продукт будет представлять собой многофункциональное кроссплатформенное приложение, которое будет иметь широкий набор функционала и мощный инструментраий по работе и анализу обрабатываемых данных. Основная функия приложения, на которую будет делаться ориентир - это предсказание значений компонентов нефти после ее обработки определенным химическим составом. То есть, будет подаваться на вход файл с параметрами нефти до обработки, следующий шаг - модель ИИ делает прецизионный подбор класса и процентного соотношения химии исходя из ее состава, так чтобы выходной продукт в виде нефти, обработанной химическим составляющим, максимально был приближен в выходу идеальной нефти в соотвествии с исследованиями. Таким образом, на выходе работы функционала, получаем файл, где есть инофрмация о нефти в первоначально состояни, после обработки и класс и процент химического состава, который был подобран. 

## Project structure

Folders:

### `Dataset:`

Содержит 1 набор данных:
 - data - переделанная и объединенная дата, которая далее в коде разбивается на вход и выход у ML алгоритма
 
### `Input_user_data:`

Содержит то, как должны выглядеть входные данные от пользователя в формате xlsx, то есть подающиеся на вход алгоритму

### `Notebooks:`

Содержит один ноутбук файл, где произодились расчеты, анализ данных, выполнение моделей машинного обучения
 
### `Saved_models:`

Содержит одну сохраненную модель сложной регрессии, которая выполляет предсказание параметров нефти после химии

### `Static:`

Содержит стили css и папку images: превью и заставку

### `Templates:`
 
Содержит файл html, главная страница приложения

## Built With

- Python
- FastAPI
- env: anaconda with Tensorflow-gpu
- jinja2
- notebooks.ipynb
- CSS,HTML

## Future Updates

- [ ] ML model optimization - MLOps
- [ ] New relevant data from other sources
- [ ] Scaling the dataset into more features
- [ ] Switch to new interface and more functionality
- [ ] Deployment 


## Author

**Ruslan Safaev, Ilsur Yaleev, Alexander Zubov**
telegram links:
- [Ruslan Safaev](https://t.me/MabelHUGO)
- [Ilsur Yaleev]( https://t.me/i_yaleev)
- [Alexander Zubov](https://t.me/dump5)

# Project Description
 ## ПРЕДСКАЗАТЕЛЬНАЯ МОДЕЛЬ ПО ПОДБОРУ ИНДИВИДУАЛЬНОЙ ХИМИИ НА ОСНОВЕ ИИ
 Проект создан для : нефтегазовых , нефтехимических и сервесных компаний
 ## ЦЕЛЬ 
 Создать универсальное приложение с интегрированным ИИ, позволяющее более качественно и быстро проводить лабораторные исследования по подбору химии на проблемные скважины
 ## ПРОЦЕСС МОДЕЛИРОВАНИЯ 
 - Изучены статьи и работы, которые могут быть как подспорье для создания гипотезы и БД для нашей задачи
 - Собрана и очищена база данных по нефти до и после применения, а также БД по химии       
 - Посчитаны погрешности применения в пласте и в лабораторных условиях       
 - Посчитана эффективность влияния активного вещества, количества вещества в смеси           
 - Подобрана архитектуры алгоритма ИИ и проведены комплексные меры тестирования по метрикам качества                                           
 - Проведены тестовые испытания на реальных данных нефти и химии, добавлены дополительные характеристики для повешения точности          
 ## РЕЗУЛЬТАТЫ ПРОЕКТА
 - Создана уникальная БД по нефти и химии      
 - Создано приложение, интегрирующее ИИ и решающее вышепоставленные задачи        
 - Взаимодействия с потенциальными заказчиками на конференциях (предложение о коллабе на CollabDay от ПАО Татнефть )    
 - Заявка проекта на финансирование за счёт грантовых средств  
 ## Работа с базами данных
 - Данные собирались и форматировались на основе анализа спектрофотометра и фурье-спектроскопии
 - Собрав все, убирались параметры не влияющие на взаимодейтсвие с химией
 ## ИСПОЛЬЗОВАНИЕ
 На данный момент проект можно использовать как код для расчетов и подборов химии на основе своих БД    
  P.s на данный момент мы ведём работу над софтом , для адаптация  ее под нужды потребителей , которым планируем сдавать в лизинг
 ## Дальнейшие планы        
 - Подбор дополнительных фичей совместно с экспертами в облисти химии нефти и газа           
 - Разработка софта , в котором будет размещено стартовое решение , также дополнительно туда будут добавляться НИОКРЫ и стартапы на основе ИИ , которые потребитель также сможет использовать 
