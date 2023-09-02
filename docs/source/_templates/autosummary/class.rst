{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. rubric:: {{ _('Methods') }}
   .. automethod:: __init__

   {% if methods %}
   .. autosummary::
   {% for item in all_methods %}
      {%- if not item.startswith('_')%}
      ~{{ name }}.{{ item }}
      {%- endif %}
   {%- endfor %}
   {% for item in all_methods %}
      {%- if item.startswith('_') and not item.startswith('__')%}
      ~{{ name }}.{{ item }}
      {%- endif %}
   {%- endfor %}
   {% endif %}

   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Properties') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% if methods %}
   .. HACK -- the point here is that we don't want this to appear in the output, but the autosummary should still generate the pages.
      .. autosummary::
         :toctree:
      {% for item in all_methods %}
         {%- if not item.startswith('__') or item in ['__call__'] %}
         {{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
{% endif %}

{% if attributes %}
   .. HACK -- the point here is that we don't want this to appear in the output, but the autosummary should still generate the pages.
      .. autosummary::
         :toctree:
      {% for item in all_attributes %}
         {%- if not item.startswith('__') %}
         {{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
{% endif %}
