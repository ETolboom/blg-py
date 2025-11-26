mod bpmn;
pub mod model_checking;
pub mod states;

use pyo3::prelude::*;
use crate::bpmn::collaboration::Collaboration;
use crate::bpmn::reader;
use crate::bpmn::reader::UnsupportedBpmnElementsError;
pub use crate::model_checking::properties::{ModelCheckingResult, Property};

pub fn run(collaboration: &Collaboration, properties: Vec<Property>) -> ModelCheckingResult {
    collaboration.explore_state_space(properties)
}

pub fn read_bpmn_from_file(file_path: &str) -> Result<Collaboration, UnsupportedBpmnElementsError> {
    reader::read_bpmn_from_file(file_path)
}

pub fn read_bpmn_from_string(
    contents: &str,
) -> Result<Collaboration, UnsupportedBpmnElementsError> {
    reader::read_bpmn_from_string(contents) // Collaboration name is irrelevant atm.
}

// PyO3 exports

#[pyclass]
#[derive(Clone)]
pub struct PyProperty {
    #[pyo3(get)]
    pub property_name: String,

    #[pyo3(get)]
    pub fulfilled: bool,

    #[pyo3(get)]
    pub problematic_elements: Vec<String>,

    #[pyo3(get)]
    pub description: String,
}

#[pymethods]
impl PyProperty {
    #[new]
    fn new(property_name: String, fulfilled: bool, problematic_elements: Vec<String>, description: String) -> Self {
        PyProperty {
            property_name,
            fulfilled,
            problematic_elements,
            description,
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "PyProperty(property_name='{}', fulfilled={}, problematic_elements={:?}, description={:?})",
            self.property_name, self.fulfilled, self.problematic_elements, self.description,
        ))
    }
}

fn type_to_description(property: &Property) -> String {
    match property {
        Property::Safeness => "The process model properly synchronizes concurrent activities.".to_string(),
        Property::OptionToComplete => "The process model can definitively reach its end state, ensuring that all started activities have a clear path to completion.".to_string(),
        Property::ProperCompletion => "There is a single unambiguous way to reach the final end event.".to_string(),
        Property::NoDeadActivities => "All activities in the process model are reachable and can be executed".to_string(),
    }
}

#[pyfunction]
fn analyze_safeness(model: &str) -> PyResult<PyProperty> {
    match read_bpmn_from_string(model) {
        Ok(collaboration) => {
            let mut property_result = run(&collaboration, vec![
                Property::Safeness,
            ]);

            let result = property_result.property_results.remove(0);

            Ok(PyProperty {
                        property_name: result.property.to_string(),
                        fulfilled: result.fulfilled,
                        problematic_elements: result.problematic_elements,
                        description: type_to_description(&result.property)
            })
        },
        Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }
}


#[pyfunction]
fn analyze_dead_activities(model: &str) -> PyResult<PyProperty> {
    match read_bpmn_from_string(model) {
        Ok(collaboration) => {
            let mut property_result = run(&collaboration, vec![
                Property::NoDeadActivities,
            ]);

            let result = property_result.property_results.remove(0);

            Ok(PyProperty {
                property_name: result.property.to_string(),
                fulfilled: result.fulfilled,
                problematic_elements: result.problematic_elements,
                description: type_to_description(&result.property)
            })
        },
        Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }
}


#[pyfunction]
fn analyze_option_to_complete(model: &str) -> PyResult<PyProperty> {
    match read_bpmn_from_string(model) {
        Ok(collaboration) => {
            let mut property_result = run(&collaboration, vec![
                Property::OptionToComplete,
            ]);

            let result = property_result.property_results.remove(0);

            Ok(PyProperty {
                property_name: result.property.to_string(),
                fulfilled: result.fulfilled,
                problematic_elements: result.problematic_elements,
                description: type_to_description(&result.property)
            })
        },
        Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }
}


#[pyfunction]
fn analyze_proper_completion(model: &str) -> PyResult<PyProperty> {
    match read_bpmn_from_string(model) {
        Ok(collaboration) => {
            let mut property_result = run(&collaboration, vec![
                Property::ProperCompletion,
            ]);

            let result = property_result.property_results.remove(0);

            Ok(PyProperty {
                property_name: result.property.to_string(),
                fulfilled: result.fulfilled,
                problematic_elements: result.problematic_elements,
                description: type_to_description(&result.property)
            })
        },
        Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }
}


// This is the module "entrypoint" for Python
#[pymodule]
fn blg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProperty>()?;
    m.add_function(wrap_pyfunction!(analyze_dead_activities, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_option_to_complete, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_proper_completion, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_safeness, m)?)?;
    
    m.add("__all__", vec![
        "PyProperty",
        "analyze_dead_activities",
        "analyze_option_to_complete",
        "analyze_proper_completion",
        "analyze_safeness",
    ])?;

    Ok(())
}