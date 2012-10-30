/**
 * Manage search related activities like concept autocomplete
 */

$(document).ready(function() {
    var $concept_form = $('#concept_form');
    $concept_form.find('input:text').typeahead({
        source: function (query, process) {
            return $.getJSON($concept_form.attr('action'), $concept_form.serialize(), function (data) {
                return process(data.results);
            });
        },
        matcher: function(item) {
            return true;
        }
    });
});
