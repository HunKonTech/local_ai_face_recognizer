# family-tree-image-export Specification

## Purpose
Lets the user export a composed family tree view as an image file (PNG, JPG or SVG) for sharing and printing.

## Requirements

### Requirement: Export the whole chart
Export SHALL render the entire composed chart (all visible persons, lines, labels and generation bands), not only the part visible on screen, with every displayed photo loaded.

#### Scenario: Large chart
- **WHEN** a chart larger than the window is exported
- **THEN** the image contains every visible person box, including ones that were scrolled off screen, with their photos

### Requirement: PNG and JPG export with scale
The user SHALL be able to export to PNG and JPG at a chosen scale (1x, 2x, 3x). JPG SHALL use a white background and a chosen quality.

#### Scenario: PNG at 2x
- **WHEN** the user exports to PNG at 2x
- **THEN** the PNG's pixel size is twice the chart's size at 1x

#### Scenario: JPG background
- **WHEN** the user exports to JPG
- **THEN** areas outside boxes are white

### Requirement: SVG export
The user SHALL be able to export to SVG. Boxes, lines and text SHALL be vector elements and photos SHALL be embedded in the file.

#### Scenario: Self-contained SVG
- **WHEN** the user exports to SVG and opens the file on another computer
- **THEN** the chart shows the photos and the names are text

### Requirement: Export without selection highlight
Exported images SHALL NOT show the interactive selection and lineage highlight. Collapse markers SHALL be shown only if the user enables them for the export.

#### Scenario: Selected person
- **WHEN** person A is selected and the chart is exported
- **THEN** A's box has its normal border in the image
