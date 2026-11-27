# family-tree-views Specification

## Purpose
Defines how a family tree view is composed (root, collapsed branches, hidden and floating persons, displayed fields per view and per person) and how views are saved, loaded and exchanged as files.

## Requirements

### Requirement: Root by person or external family code
A view SHALL have a root given either as a person or as an external family code `#root#`. For a code, the root SHALL be the person whose external family code is exactly `#root#`. When no such person exists, the user SHALL be asked to choose among persons whose external code starts with `#root#`.

#### Scenario: Root from code
- **WHEN** a view is created with root `#Zoli#` and person Z has external code `#Zoli#`
- **THEN** the view is rooted at Z

#### Scenario: No exact root
- **WHEN** no person has external code `#Zoli#` but persons with `#Zoli#1` and `#Zoli#2` exist
- **THEN** the user is asked to choose the root from those persons

### Requirement: Collapse and expand branches
Collapsing a person SHALL hide their descendants, except persons that remain reachable from the root through another visible path. It SHALL keep the collapsed person's spouses visible. The collapsed person SHALL carry a marker that expands the branch again.

#### Scenario: Collapse keeps spouse
- **WHEN** person P with spouse S and children C1, C2 is collapsed
- **THEN** C1 and C2 and their descendants are hidden
- **AND** P and S remain visible and P shows an expand marker

#### Scenario: Reachable another way
- **WHEN** P is collapsed and C1 is also the child of visible person Q who is not collapsed
- **THEN** C1 remains visible

#### Scenario: Expand
- **WHEN** the user expands P
- **THEN** C1, C2 and their descendants are visible again, except individually hidden persons

### Requirement: Hide individual persons
The user SHALL be able to hide any single person, including a spouse, from the view. A hidden person SHALL also hide persons reachable only through them. The view SHALL list hidden persons and allow showing each again.

#### Scenario: Hide a spouse
- **WHEN** the user hides spouse S of P
- **THEN** S is not drawn and P remains visible

#### Scenario: Show again
- **WHEN** the user selects S in the hidden-persons list and shows them
- **THEN** S is drawn next to P again

### Requirement: Displayed fields per view
A view SHALL define an ordered list of person fields (built-in and custom) shown in every person box below the name. A person with no value for a field SHALL show no line for it.

#### Scenario: Occupation shown
- **WHEN** the view shows nickname and "Foglalkozás", and person A has nickname "Pisti" and occupation "tanító"
- **THEN** A's box shows "Pisti" and "tanító" below the name

#### Scenario: Empty value
- **WHEN** person B has no occupation
- **THEN** B's box has no occupation line

### Requirement: Field display override per person
The user SHALL be able to show or hide individual fields for a single person in the view, overriding the view's field list for that person only. The override SHALL be part of the saved view.

#### Scenario: Hide one person's dates
- **WHEN** the view shows dates and the user hides dates for person A
- **THEN** A's box shows no dates and other boxes still show theirs

#### Scenario: Extra field for one person
- **WHEN** the view does not show "Foglalkozás" and the user shows it for person A
- **THEN** only A's box shows the occupation

### Requirement: Box height follows content per generation
Every box in a generation row SHALL be tall enough for the box with the most displayed lines in that row. Rows SHALL NOT overlap.

#### Scenario: One tall box
- **WHEN** one person in generation 2 shows 5 lines and the others show 2
- **THEN** all generation-2 boxes have the 5-line height and generation 3 starts below them

### Requirement: Save and load views
Views SHALL be saved in the database under a unique name, capturing the root, collapsed, hidden and floating persons, the field list, per-person overrides and display options. Loading a view SHALL restore that state on current data. Views SHALL be renamable, duplicable and deletable.

#### Scenario: Round trip
- **WHEN** the user saves a view "Nagyszülők", closes Tree Studio, reopens it and loads "Nagyszülők"
- **THEN** the same persons are collapsed, hidden and floating, and the same fields are shown

#### Scenario: Person deleted since save
- **WHEN** a hidden person of a saved view was deleted in Face-Local and the view is loaded
- **THEN** the view loads and the deleted person is ignored

### Requirement: Exchange views as JSON files
A view SHALL be exportable to a JSON file and importable from one. Each referenced person SHALL carry identifying data (id, name, family codes, birth date). On import, persons SHALL be matched by id when that data agrees, otherwise by the identifying data, and unmatched references SHALL be reported and dropped.

#### Scenario: Import in same database
- **WHEN** a view exported to JSON is imported into the same database
- **THEN** the imported view equals the original

#### Scenario: Import with unknown persons
- **WHEN** a JSON view references 2 persons that do not exist in the database
- **THEN** the import succeeds and reports the 2 unmatched persons
