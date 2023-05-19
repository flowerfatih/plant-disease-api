treatment_links = {
"Apple Scab": "https://www2.gov.bc.ca/gov/content/industry/agriculture-seafood/animals-and-crops/plant-health/insects-and-plant-diseases/tree-fruits/apple-scab",
"Apple Black Rot": "https://extension.umn.edu/plant-diseases/black-rot-apple",
"Apple Cedar Rust": "https://extension.umn.edu/plant-diseases/cedar-apple-rust",
"Apple Healthy": "https://www.almanac.com/plant/apples",
"Blueberry Healthy": "https://extension.umn.edu/fruit/growing-blueberries-home-garden",
"Cherry Healthy": "https://www.almanac.com/plant/cherries",
"Cherry Powdery Mildew": "https://treefruit.wsu.edu/crop-protection/disease-management/cherry-powdery-mildew/",
"Corn Cercospora": "https://extension.umn.edu/corn-pest-management/gray-leaf-spot-corn",
"Corn Common Rust": "https://extension.umn.edu/corn-pest-management/common-rust-corn",
"Corn Healthy": "https://www.almanac.com/plant/corn",
"Corn Northern Leaf Blight": "https://extension.umn.edu/corn-pest-management/northern-corn-leaf-blight",
"Grape Black Rot": "https://www.gardeningknowhow.com/edible/fruits/grapes/black-rot-grape-treatment.htm",
"Grape Esca": "https://grapes.extension.org/grapevine-measles/",
"Grape Healthy": "https://www.rhs.org.uk/fruit/grapes/grow-your-own",
"Grape Leaf Blight": "https://www.planthealthaustralia.com.au/wp-content/uploads/2013/11/Bacterial-blight-of-grapevine-FS.pdf",
"Orange Haunglongbing": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8636133/",
"Peach Bacterial Spot": "https://www.aces.edu/blog/topics/crop-production/bacterial-spot-treatment-in-peaches/",
"Peach Healthy": "https://www.masterclass.com/articles/how-to-grow-a-peach-tree-in-your-backyard",
"Pepper Bacterial Spot": "https://extension.wvu.edu/lawn-gardening-pests/plant-disease/fruit-vegetable-diseases/bacterial-leaf-spot-of-pepper",
"Pepper Healthy": "https://bonnieplants.com/blogs/how-to-grow/growing-peppers",
"Potato Early Blight": "https://www.planetnatural.com/pest-problem-solver/plant-disease/early-blight/",
"Potato Healthy": "https://www.gardendesign.com/vegetables/potatoes.html",
"Potato Late Blight": "https://www.planetnatural.com/pest-problem-solver/plant-disease/late-blight/",
"Raspberry Healthy": "https://www.rhs.org.uk/fruit/raspberries/grow-your-own",
"Soybean Healthy": "https://www.rhs.org.uk/vegetables/soya-beans/grow-your-own",
"Squash Powdery Mildew": "https://www.gardeningknowhow.com/edible/vegetables/squash/powdery-mildew-in-squash.htm",
"Strawberry Healthy": "https://www.rhs.org.uk/fruit/strawberries/grow-your-own",
"Strawberry Leaf Scorch": "https://www.gardeningknowhow.com/edible/fruits/strawberry/strawberries-with-leaf-scorch.htm",
"Tomato Bacterial Spot": "https://extension.umn.edu/disease-management/bacterial-spot-tomato-and-pepper",
"Tomato Early Blight": "https://extension.umn.edu/disease-management/early-blight-tomato-and-potato",
"Tomato Healthy": "https://www.rhs.org.uk/vegetables/tomatoes/grow-your-own",
"Tomato Late Blight": "https://extension.wvu.edu/lawn-gardening-pests/plant-disease/fruit-vegetable-diseases/late-blight-tomatoes",
"Tomato Leaf Mold": "https://www.lovethegarden.com/uk-en/article/tomato-leaf-mould",
"Tomato Septoria Leaf Spot": "https://extension.wvu.edu/lawn-gardening-pests/plant-disease/fruit-vegetable-diseases/septoria-leaf-spot",
"Tomato Spider Mites": "https://pnwhandbooks.org/insect/vegetable/vegetable-pests/hosts-pests/tomato-spider-mite",
"Tomato Target Spot": "https://apps.lucidcentral.org/ppp/text/web_full/entities/tomato_target_spot_163.htm",
"Tomato Mosaic Virus": "https://www.almanac.com/pest/mosaic-viruses",
"Tomato Yellow Leaf Curl Virus": "https://plantix.net/en/library/plant-diseases/200036/tomato-yellow-leaf-curl-virus",
}

class_names = ["Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy", "Blueberry Healthy", "Cherry Healthy", "Cherry Powdery Mildew",
               "Corn Cercospora", "Corn Common Rust", "Corn Healthy", "Corn Northern Leaf Blight", "Grape Black Rot", "Grape Esca", "Grape Healthy", "Grape Leaf Blight",
               "Orange Haunglongbing", "Peach Bacterial Spot", "Peach Healthy", "Pepper Bacterial Spot", "Pepper Healthy", "Potato Early Blight", "Potato Healthy", "Potato Late Blight",
               "Raspberry Healthy", "Soybean Healthy", "Squash Powdery Mildew", "Strawberry Healthy", "Strawberry Leaf Scorch", "Tomato Bacterial Spot", "Tomato Early Blight",
               "Tomato Healthy", "Tomato Late Blight", "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites", "Tomato Target Spot", "Tomato Mosaic Virus",
               "Tomato Yellow Leaf Curl Virus"]

model_path = "C:/Users/fthsl/OneDrive/Masaüstü/disease-api/model/resnet_model.h5"

requests_count = {}

response_dict = {"plantHealthModels":[]}