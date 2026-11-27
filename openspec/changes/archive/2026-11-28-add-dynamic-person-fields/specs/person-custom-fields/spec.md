# Spec Delta

## Purpose

Lets users extend person records with their own named fields (for example occupation), reuse each field across persons, and expose built-in and custom fields uniformly to views such as the family tree.

## ADDED Requirements

### Requirement: Define a custom field from the person editor
The person editor SHALL let the user create a new custom field by giving a non-empty label and a type (`text`, `long_text` or `date`), and enter that person's value at the same time. Labels SHALL be unique, ignoring case and surrounding whitespace.

#### Scenario: New field with a value
- **WHEN** the user adds a new field "Foglalkozás" of type text with the value "tanító" for person A and saves
- **THEN** a field definition "Foglalkozás" exists
- **AND** person A's value for it is "tanító"

#### Scenario: Duplicate label
- **WHEN** the user tries to create a field " foglalkozás " while "Foglalkozás" already exists
- **THEN** no second definition is created
- **AND** the editor offers the existing "Foglalkozás" field instead

### Requirement: Reuse existing fields for other persons
Once a field is defined, the person editor of any person SHALL offer it in a list of existing fields, so the user can add a value without defining the field again. The value input SHALL suggest values already used for that field by other persons.

#### Scenario: Pick existing field
- **WHEN** "Foglalkozás" exists and the user opens person B's editor
- **THEN** "Foglalkozás" can be chosen from the existing-field list
- **AND** after entering "orvos" and saving, person B's value is "orvos"

#### Scenario: Value suggestions
- **WHEN** person A has "tanító" for "Foglalkozás" and the user types "ta" into person B's "Foglalkozás" value
- **THEN** "tanító" is suggested

### Requirement: Editor shows only filled fields
The person editor SHALL show the custom fields that have a value for the person, in definition order. Clearing a value and saving SHALL remove that person's value and SHALL NOT delete the definition.

#### Scenario: Unfilled fields hidden
- **WHEN** person C has no custom values and the editor opens
- **THEN** no custom field inputs are shown, only the control to add one

#### Scenario: Clear a value
- **WHEN** the user empties person A's "Foglalkozás" and saves
- **THEN** person A has no value for "Foglalkozás"
- **AND** the "Foglalkozás" definition still exists

### Requirement: Same save behaviour from every entry point
Wherever the person editor is opened (Persons tab, image browser, family tree, sidebar), saving SHALL persist custom field changes identically to built-in fields. Cancelling SHALL persist nothing.

#### Scenario: Edit from family tree
- **WHEN** the user edits person A from the family tree panel, sets "Foglalkozás" to "lakatos" and saves
- **THEN** the Persons tab editor for person A shows "lakatos"

#### Scenario: Cancel
- **WHEN** the user adds a new field and value and then cancels the editor
- **THEN** neither the definition nor the value is stored

### Requirement: Manage field definitions
The system SHALL let the user rename a field, change the field order, and delete a field. Renaming SHALL keep all values. Deleting SHALL require confirmation that states how many persons have a value, and SHALL remove the definition and all its values.

#### Scenario: Rename keeps values
- **WHEN** "Foglalkozás" is renamed to "Szakma"
- **THEN** person A's value "tanító" appears under "Szakma"

#### Scenario: Delete with confirmation
- **WHEN** the user deletes "Szakma", which 2 persons use
- **THEN** the confirmation mentions 2 persons
- **AND** after confirming, the field and both values are gone

### Requirement: Values follow the person on merge and delete
Merging persons SHALL keep the target's custom values and copy the source's values for fields the target has no value for. Undoing the merge SHALL restore the source's custom values. Deleting a person SHALL delete its custom values.

#### Scenario: Merge fills gaps
- **WHEN** source has "Foglalkozás"="tanító" and "Hobbi"="horgászat", target has "Foglalkozás"="igazgató", and the source is merged into the target
- **THEN** the target has "Foglalkozás"="igazgató" and "Hobbi"="horgászat"

#### Scenario: Undo merge
- **WHEN** that merge is undone
- **THEN** the restored source again has "Foglalkozás"="tanító" and "Hobbi"="horgászat"

#### Scenario: Delete person
- **WHEN** a person with custom values is deleted
- **THEN** no value rows for that person remain

### Requirement: Field catalog with stable keys
The system SHALL provide a catalog that lists built-in person fields and custom fields, each with a stable key, a display label and a type, and SHALL return a person's values by key. Built-in keys SHALL NOT change when labels are translated. Custom keys SHALL NOT change when the field is renamed.

#### Scenario: Catalog contents
- **WHEN** "Foglalkozás" is defined and the catalog is requested
- **THEN** it contains the built-in nickname field and the custom "Foglalkozás" field, each with a key

#### Scenario: Key survives rename
- **WHEN** "Foglalkozás" is renamed to "Szakma"
- **THEN** the catalog key of that field is unchanged

### Requirement: Custom fields in person export
The person data export SHALL include each person's custom field values with their current labels.

#### Scenario: Export
- **WHEN** person A with "Foglalkozás"="tanító" is exported
- **THEN** the exported person record contains "Foglalkozás" with "tanító"

### Requirement: Older databases keep working
Opening a database created before this change SHALL add the custom field storage automatically and SHALL leave existing person data unchanged.

#### Scenario: Upgrade
- **WHEN** an existing database without custom field tables is opened
- **THEN** the app starts, persons keep all built-in values, and custom fields can be added
