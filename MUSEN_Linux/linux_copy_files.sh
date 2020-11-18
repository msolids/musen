#!/bin/bash

DIRS=(
Databases/AgglomerateDatabase
Databases/GeometriesDatabase
Databases/MaterialsDatabase
GeneralFunctions
GeneralFunctions/MUSEN
GeneralFunctions/SimResultsStorage
MUSEN/Models/ExternalForce/ViscousField
MUSEN/Models/LiquidBonds/CapilaryViscous
MUSEN/Models/ParticleParticle/Hertz
MUSEN/Models/ParticleParticle/HertzMindlin
MUSEN/Models/ParticleParticle/HertzMindlinLiquid
MUSEN/Models/ParticleParticle/JKR
MUSEN/Models/ParticleParticle/LinearElastic
MUSEN/Models/ParticleParticle/PopovJKR
MUSEN/Models/ParticleParticle/SimpleViscoElastic
MUSEN/Models/ParticleParticle/TestSinteringModel
MUSEN/Models/ParticleWall/PWHertzMindlin
MUSEN/Models/ParticleWall/PWHertzMindlinLiquid
MUSEN/Models/ParticleWall/PWJKR
MUSEN/Models/ParticleWall/PWPopovJKR
MUSEN/Models/ParticleWall/PWSimpleViscoElastic
MUSEN/Models/SolidBonds/BondModelElastic
MUSEN/Models/SolidBonds/BondModelKelvin
MUSEN/Models/SolidBonds/BondModelLinearPlastic
MUSEN/Models/SolidBonds/BondModelThermal
MUSEN/Modules/BondsGenerator
MUSEN/Modules/ContactAnalyzers
MUSEN/Modules/FileManager
MUSEN/Modules/Geometries
MUSEN/Modules/ObjectsGenerator
MUSEN/Modules/PackageGenerator
MUSEN/Modules/ResultsAnalyzer
MUSEN/Modules/ScriptInterface
MUSEN/Modules/SimplifiedScene
MUSEN/Modules/Simulator
MUSEN/Modules/SystemStructure
QTDialogs/AgglomeratesDatabaseTab
QTDialogs/BondsGeneratorTab
QTDialogs/GeneralMusenDialog
QTDialogs/GeometriesDatabaseTab
QTDialogs/GeometriesEditorTab
QTDialogs/MaterialsDatabaseTab
QTDialogs/MUSENConfigurationTab
QTDialogs/MUSENMainWindow
QTDialogs/ObjectsEditorTab
QTDialogs/ObjectsGeneratorTab
QTDialogs/PackageGeneratorTab
QTDialogs/QtOpenGLView
QTDialogs/QtWidgets
QTDialogs/ResultsAnalyzerTab
QTDialogs/SampleAnalyzerTab
QTDialogs/SceneEditorTab
QTDialogs/SceneInfoTab
QTDialogs/SimulatorTab
MUSEN/BuildVersion
MUSEN/CMusen
MUSEN/QT_GUI
MUSEN/QT_GUI/Pictures
MUSEN/QT_GUI/QT_GUI
MUSEN/QT_GUI/Resources
MUSEN/QT_GUI/shaders
MUSEN/QT_GUI/styles
)

rm -rf ./MUSEN_src

(cd ../MUSEN/BuildVersion && chmod +x generate_build_version.sh && ./generate_build_version.sh)

for DIR in "${DIRS[@]}"; do 
	mkdir -p ./MUSEN_src/$DIR
	find $PWD/../$DIR -maxdepth 1 -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.proto' -o -name '*.cuh' -o -name '*.cu' -o -name '*.ui' -o -name '*.png' -o -name '*.glsl' -o -name '*.qss' -o -name '*.qrc' -o -name '*.sh' \) -exec cp '{}' ''$PWD'/MUSEN_src/'$DIR'' ';'
done

cp ${PWD}/../LICENSE ${PWD}/MUSEN_src/