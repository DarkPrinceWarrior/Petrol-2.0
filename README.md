<h1 align="center"><Petrol prediction></h1>

<p align="center"><project-description></p>

## Screenshots

![Home Page](/static/images/preview.jpg "Home Page")

![](/screenshots/2.png)
 
  
## Work process
Программа работает следующим образом: подается на вход xlsx файл с параметрами нефти до обработки, потом модель подбирает за счет алгоитма ML лучший класс и процент химии этого класса, так чтобы выход максимально был приближен в выходу идеальной нефти, таким образом получаем csv файл, где к входному датасету добавились класс и процент химии

## Project structure

Folders:

### `Dataset:`

Содержит 2 набора данных:
 - original - оригинальняа дата, которая неструктурирована, по сути все данные с установки
 - data_new - переделанная и объединенная дата, которая далее в коде разбивается на вход и выход у ML алгоритма
 
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
 Создать алгоритм искусственного интеллекта позволяющий более качественно и быстро проводить лабораторные исследования по подбору химии на проблемные скважины
 ## ПРОЦЕСС МОДЕЛИРОВАНИЯ 
 - Собрана и очищена база данных по нефти и химии , до и после применения       
 - Посчитаны погрешности применения в пласте и в лабораторных условиях       
 - Посчитана эффективность влияния активного вещества , количества вещества в смеси           
 - Подобрана ML                                            
 - Проведены тестовые испытания и добавлены дополительные характеристики для повешения точности         
 ## РЕЗУЛЬТАТЫ ПРОЕКТА
 - Создана уникальная тренировочная БД по нефти и Химии      
 - Создан алгоритм ИИ решающий вышепоставленные задачи        
 - Взаимодействия с потенциальными заказчиками на конференциях ( предложение о коллабе на CollabDay от ПАО Татнефть )    
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