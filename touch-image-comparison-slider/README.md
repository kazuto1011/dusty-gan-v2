# Cndk.BeforeAfter.Js v.0.0.2

It is a very simple jQuery plugin that will meet your needs.

![Screenshot](https://i.ibb.co/9bCgWzc/Ekran-Resmi-2020-02-29-21-36-37.png)

## Demo

See the demo content of the data from the link.

[Go to Demo](https://www.ilkerc.com/cndkbeforeafterdemo/demo.html)

## Usage

#### Import Javascript Files

```html
<script src="https://code.jquery.com/jquery-3.4.1.min.js"></script>
<script src="cndk.beforeafter.js"></script>
```

#### Import CSS File

```html
<link href="cndk.beforeafter.css" rel="stylesheet">
```

#### HTML Code

```html
<div class="beforeimagetest">
   <div data-type="data-type-image">
      <div data-type="before"><img src="images/a1.jpg"></div>
      <div data-type="after"><img src="images/a2.jpg"></div>
   </div>
</div>
```

#### Javascript Run Code

```html
$(".beforeimagetest").cndkbeforeafter(
    {
       showText: false,
       seperatorColor: "blue",
       seperatorWidth: "1px",
       hoverEffect: false,
       beforeText: "BEFORE",
       afterText: "TEXT"
    }
);
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)