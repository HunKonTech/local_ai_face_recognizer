# Spec Delta

## Purpose

A separate desktop app for composing family tree charts that works on the same database and person data as Face-Local, including editing persons and relationships and adding persons by drag and drop.

## ADDED Requirements

### Requirement: Start on the shared database
Tree Studio SHALL open the same database and configuration as Face-Local, honour the same database override option, and SHALL be startable on its own or from a Face-Local menu entry. Both apps SHALL be able to run at the same time on one database.

#### Scenario: Launch from Face-Local
- **WHEN** the user chooses the Tree Studio menu entry in Face-Local
- **THEN** Tree Studio opens as a separate window on the same database
- **AND** Face-Local stays usable

#### Scenario: Database override
- **WHEN** Tree Studio is started with a database path override
- **THEN** it opens that database

### Requirement: Searchable person list
Tree Studio SHALL show a searchable list of persons with name and thumbnail, from which persons can be dragged onto the tree. Persons already visible in the current view SHALL be marked as such.

#### Scenario: Search
- **WHEN** the user types "Kiss" into the search field
- **THEN** only persons whose name matches "Kiss" are listed

### Requirement: Drop a related person
When a person who is related to the current tree (reachable from its root in the unfiltered family graph) is dropped onto the tree, Tree Studio SHALL show that person in their place, expand every collapsed ancestor on the way, and un-hide them. It SHALL NOT create any relationship.

#### Scenario: Hidden grandchild
- **WHEN** grandchild G is hidden and their parent P is collapsed, and the user drops G onto the tree
- **THEN** P is expanded and G is visible under P
- **AND** no relationship is created

### Requirement: Drop an unrelated person on a person
When a person not related to the current tree is dropped onto a person box, Tree Studio SHALL ask which relationship to create (parent, child, spouse, or sibling when the target has parents) and SHALL create it in the shared database. Cancelling SHALL change nothing.

#### Scenario: New spouse
- **WHEN** unrelated person U is dropped on person A and the user picks spouse
- **THEN** a spouse relationship between A and U exists
- **AND** U appears next to A in the view and in Face-Local's family tree after refresh

#### Scenario: Sibling unavailable
- **WHEN** U is dropped on person R who has no parents
- **THEN** the sibling choice is not offered

### Requirement: Drop an unrelated person on empty space
When a person not related to the current tree is dropped onto empty space, Tree Studio SHALL add them to the view as a floating person without creating a relationship, and SHALL offer a Connect action that links them to a chosen person with a chosen relationship.

#### Scenario: Floating then connect
- **WHEN** U is dropped on empty space
- **THEN** U is shown outside the tree with no relationship created
- **AND** after Connect with person A as child, U is shown under A and is no longer floating

### Requirement: Edit persons as in Face-Local
Tree Studio SHALL open the same person editor as Face-Local, including custom fields, and save through the same rules. It SHALL open the same relationship editor. After saving, the view SHALL reflect the change.

#### Scenario: Edit nickname
- **WHEN** the user double-clicks person A, changes the nickname and saves
- **THEN** A's box shows the new nickname if the nickname field is displayed
- **AND** Face-Local shows the new nickname after refresh

### Requirement: Detect changes made in Face-Local
Tree Studio SHALL detect, on window activation, changes written by other processes (including Face-Local) and offer or perform a refresh as Face-Local does, honouring the same automatic refresh setting.

#### Scenario: Person renamed in Face-Local
- **WHEN** person A is renamed in Face-Local and Tree Studio's window is activated
- **THEN** Tree Studio shows the external change notice or, with auto refresh on, the new name
