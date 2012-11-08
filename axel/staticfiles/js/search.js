/**
 * Manage search related activities like concept autocomplete
 */

$(document).ready(function() {
    var $concept_form = $('#concept_form');

    // setup concept autocomplete
    $concept_form.find('input:text').typeahead({
        source: function (query, process) {
            return $.getJSON($concept_form.attr('action'), $concept_form.serialize(), function (data) {
                return process(data.results);
            });
        },
        matcher: function(item) {
            return true;
        },
        updater: function(item) {
            // Update list of concepts
            var $container = $('#selected_concepts');
            var $label = $container.find('.label.hidden').clone();
            item = item.replace(new RegExp('\\s(\\d+)$'), function ($1, match) {
                return ''
            });
            $label.removeClass('hidden').text(item).append('<button class="close">×</button>');
            $container.append($label);
        },
        highlighter: function (item) {
            var query = this.query.replace(/[\-\[\]{}()*+?.,\\\^$|#\s]/g, '\\$&');
            // additionally parse score
            item = item.replace(new RegExp('(' + query + ')', 'ig'), function ($1, match) {
                return '<strong>' + match + '</strong>'
            });
            return item.replace(new RegExp('\\s(\\d+)$'), function ($1, match) {
                return ' <span class="badge badge-success">' + match + '</span>'
            });
        }
    });

    // setup show articles function
    $('#show_articles').on('click', function() {
        var concepts = $.map($('#selected_concepts').find('span.label:not(.hidden)'), function(item) {
            $('#concept_form').find('div.search').append($('<input name="concepts" type="hidden" value="'+$(item).text().slice(0, -1)+'">'));
            return $(item).text().slice(0, -1);
        });
        $.post($(this).attr('data-url'), $('#concept_form').serialize(), function(data){
            // show articles
            $('#search_results').empty().append(data);
        });
    });
});
