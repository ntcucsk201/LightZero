function BaseBoard(row, col, name, mode, scale = 1) {
  if (name == null) return;
  this.name = name;
  this.piece_row = row;
  this.piece_col = col;
  this.blank_row = row;
  this.blank_col = col;

  this.boundary_top = 0;
  this.boundary_left = 0;

  this.blank_size = 40 * scale;
  this.border_width = 3 * scale;
  this.piece_font_size = 30 * scale;
  this.piece_bound_top = 0;
  this.piece_bound_left = 0;
  this.bound_font_size = 20 * scale;
  this.die_blank_size = 32 * scale;
  this.die_piece_font_size = 24 * scale;
  this.movelist_font_size = 12 * scale;

  this.div_board = this.new_div("div_board_" + name);
  this.div_piece = this.new_div("div_piece_" + name);
  this.div_board = this.new_div("div_board_" + name);
  this.div_movelist = this.new_div("div_movelist_" + name);
  this.div_red_died = this.new_div("div_red_died" + name);

  if (name == "EC") {
    this.div_blue_died = this.new_div("div_blue_died" + name);
  } else {
    this.div_black_died = this.new_div("div_black_died" + name);
  }

  this.div_endgame_movelist = this.new_div("div_endgame_movelist_" + name);
  this.div_path = null;

  this.LANG_EN = 0;
  this.LANG_CH = 1;
  this.LANG = this.LANG_CH;

  this.STR_NUM = ["０", "１", "２", "３", "４", "５", "６", "７", "８", "９"];
  this.EN_NUM = ["", "ａ", "ｂ", "ｃ", "ｄ"];
  this.CH_NUM = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九"];


  /*test*/
  this.record_genmove = ""; // if isstopboard == true && recv gnemove, record genmvoe
  this.isstopboard = false;
  this.his_create_name = false;
  this.his_openf = "";
  this.his_step = 0;  // keydown 
  this.metaChar = false;  // keydown
  this.exampleKey = 16; // keydown
  this.split_str = "\r\n"; // windows->\r\n, linux->\n
  this.os_type = "\r\n";  // windows->\r\n, linux->\n
  this.turn = 0; // 1 my turn 0 opp
  this.board_data = ""; // board information
  this.step = -1;
  this.info_row = new Array();  // user information row 
  this.info_error_row = new Array();  // user error information row
  this.info = new Array();  // user information
  this.info_error = new Array();  // user error information
  this.acttime = new Array(); // board history user time
  this.actlist = new Array(); // board history user move
  this.tmp = null;
  this.player = 0;
  this.player_name = new Array(); // 0: players 1: player1 2:player2
  this.socket = null;
  this.clr = -1; // 0: red, 1: black
  this.mode = mode; // 1: index.html, 0: history, 2: abort 3: observation 4: contest_history
  this.highlight = false;
  this.initBoard = new Array(); // record init board
  this.initRemainDarkChess = new Array(); // record init remain darkchess
  this.PCE_QUA = new Map([
    ["K", 1],
    ["G", 2],
    ["M", 2],
    ["R", 2],
    ["N", 2],
    ["C", 2],
    ["P", 5],
    ["k", 1],
    ["g", 2],
    ["m", 2],
    ["r", 2],
    ["n", 2],
    ["c", 2],
    ["p", 5]
  ]);
  this.PCE_DIE = new Map([
    ["K", 0],
    ["G", 0],
    ["M", 0],
    ["R", 0],
    ["N", 0],
    ["C", 0],
    ["P", 0],
    ["k", 0],
    ["g", 0],
    ["m", 0],
    ["r", 0],
    ["n", 0],
    ["c", 0],
    ["p", 0]
  ]);
  this.PCE_MAP = {
    "E": { en: "", ch: "", clr: "EMPTY" },
    "K": { en: "K", ch: "帥", clr: "RED" },
    "G": { en: "G", ch: "仕", clr: "RED" },
    "M": { en: "M", ch: "相", clr: "RED" },
    "R": { en: "R", ch: "俥", clr: "RED" },
    "N": { en: "N", ch: "傌", clr: "RED" },
    "C": { en: "C", ch: "炮", clr: "RED" },
    "P": { en: "P", ch: "兵", clr: "RED" },
    "D": { en: "", ch: "", clr: "DARK" },
    "k": { en: "k", ch: "將", clr: "BLACK" },
    "g": { en: "g", ch: "士", clr: "BLACK" },
    "m": { en: "m", ch: "象", clr: "BLACK" },
    "r": { en: "r", ch: "車", clr: "BLACK" },
    "n": { en: "n", ch: "馬", clr: "BLACK" },
    "c": { en: "c", ch: "包", clr: "BLACK" },
    "p": { en: "p", ch: "卒", clr: "BLACK" },
  };
  this.PCE_TYPE = {
    "DARK": { className: "piece_dark" },
    "RED": { className: "piece_red" },
    "BLACK": { className: "piece_black" },
    "DARK_RED": { className: "piece_dark_red" },
    "DARK_BLACK": { className: "piece_dark_black" },
  };
}

BaseBoard.prototype = {
  new_div: function (id) {
    document.write("<div id=\"" + id + "\"></div>");
    return document.getElementById(id);
  },
  new_div_shape: function (className, top, left, height, width, font_size, t_index, css_text = "") {
    var div = document.createElement("div");
    div.className = className;
    if (t_index != null) { div.id = t_index; }
    //if(t_index != 0){div.id = t_index;}
    div.style.cssText = "width:" + width + "px;";
    div.style.cssText += "height:" + height + "px;";
    div.style.cssText += "top:" + top + "px;";
    div.style.cssText += "left:" + left + "px;";
    div.style.cssText += "line-height:" + height + "px;";
    div.style.cssText += "font-size:" + font_size + "px;";
    div.style.cssText += css_text;
    return div;
  },
  new_div_shape_line_height: function (className, top, left, height, width, font_size, line_height) {
    var div = document.createElement("div");
    div.className = className;
    div.style.cssText = "width:" + width + "px";
    div.style.cssText += "height:" + height + "px";
    div.style.cssText += "top:" + top + "px";
    div.style.cssText += "left:" + left + "px";
    div.style.cssText += "line-height:" + line_height + "px;";
    div.style.cssText += "font-size:" + font_size + "px;";
    return div;
  },

  new_board_shape: function (className, top, col) {
    return this.new_div_shape(className, row, col, this.blank_size, this.piece_font_size);
  },

  new_board_blank: function (row, col, text) {
    var top = row * (this.blank_size + this.border_width);
    var left = col * (this.blank_size + this.border_width);
    var s = null;
    if (text == "CC_blank") {
      var test_top = top - 20;
      var test_left = left - 20;
      s = row + "_" + col;
      var obj = this.new_div_shape("CC_blank", test_top, test_left, this.blank_size, this.blank_size, this.piece_font_size, s);
      obj.onclick = this.shape_mouse_click;
    } else {
      switch (this.name) {
        case "CC":
          var obj = this.new_div_shape("blank", top, left, this.blank_size, this.blank_size, this.piece_font_size);
          break;
        case "CDC":
          var cols = String.fromCharCode(this.EN_NUM[col].charCodeAt() - 65248);
          var rows = String.fromCharCode(this.STR_NUM[9 - row].charCodeAt() - 65248);
          s = cols + "_" + rows;
          var obj = this.new_div_shape("blank", top, left, this.blank_size, this.blank_size, this.piece_font_size, s);
          obj.onclick = this.shape_mouse_click;
          break;
        case "MS":
          var cols = 6 - col;
          var rows = String.fromCharCode(this.EN_NUM[row].charCodeAt());
          s = cols + "_" + rows;
          var obj = this.new_div_shape("CC_blank", top, left, this.blank_size, this.blank_size, this.piece_font_size, s);
          obj.addEventListener('click', this.shape_mouse_click, false);

          //obj.onclick = this.shape_mouse_click;
          break;
      }
    }
    /*else if( this.name == "CDC" )
    {
      var cols = String.fromCharCode(this.EN_NUM[col].charCodeAt() - 65248);
      var rows = String.fromCharCode(this.STR_NUM[9-row].charCodeAt() - 65248);
      s = cols + "_" + rows;
      var obj = this.new_div_shape("blank", top, left, this.blank_size, this.blank_size, this.piece_font_size, s);
    }
    else if( this.name == "MS" )
    {

      var cols = 6 - col;
      var rows = String.fromCharCode(this.EN_NUM[row].charCodeAt() - 65248);
      s = cols + "_" + rows;

      var obj = this.new_div_shape("CC_blank", top, left, this.blank_size, this.blank_size, this.piece_font_size, s);
    }
    else
    {
      var obj = this.new_div_`shape("blank", top, left, this.blank_size, this.blank_size, this.piece_font_size);
    }*/
    /* mouse click */


    /*if( (this.name == "CC" && text == "CC_blank") )
    {
      obj.onclick = this.shape_mouse_click;
      //console.log("mouse click");
    }
    else if ( this.name == "CDC" ){obj.onclick = this.shape_mouse_click;}*/
    return obj;
  },
  new_slash: function (row, col) {
    var top = row * (this.blank_size + this.border_width);
    var left = col * (this.blank_size + this.border_width);
    return this.new_div_shape("slash", top, left, this.blank_size, this.blank_size, this.piece_font_size);
  },
  new_backslash: function (row, col) {
    var top = row * (this.blank_size + this.border_width);
    var left = col * (this.blank_size + this.border_width);
    return this.new_div_shape("backslash", top, left, this.blank_size, this.blank_size, this.piece_font_size);
  },
  new_river: function (row, col) {
    var top = row * (this.blank_size + this.border_width);
    var left = col * (this.blank_size + this.border_width);
    var width = (this.blank_col) * (this.blank_size + this.border_width) - this.border_width;
    return this.new_div_shape("river", top, left, this.blank_size, width, this.piece_font_size);
  },
  new_board_piece: function (row, col, className, text) {
    var top = row * (this.blank_size + this.border_width);
    var left = col * (this.blank_size + this.border_width);
    var div = null;
    var piece_blank = null;
    switch (this.name) {
      case "CC":
        div = this.new_div_shape("piece", 0, 0, this.blank_size, this.blank_size, this.piece_font_size);
        var piece_str = this.new_div_shape(className, -2, -2, this.blank_size, this.blank_size, this.piece_font_size);
        piece_str.append(text);
        div.append(piece_str);
        div.onclick = this.piece_click;
        piece_blank = document.getElementById(row + "_" + col);
        piece_blank.appendChild(div);
        return piece_blank;
      case "CDC":
        div = this.new_div_shape("piece", -3, -3, this.blank_size, this.blank_size, this.piece_font_size);
        var piece_str = this.new_div_shape(className, -2, -2, this.blank_size, this.blank_size, this.piece_font_size);
        piece_str.append(text);
        div.append(piece_str);
        var r = (9 - row);
        div.onclick = this.piece_click;
        var cols = String.fromCharCode(this.EN_NUM[col].charCodeAt() - 65248);
        var rows = String.fromCharCode(this.STR_NUM[r].charCodeAt() - 65248);
        piece_blank = document.getElementById(cols + "_" + rows);
        piece_blank.appendChild(div);
        return piece_blank;
      case "MS":
        var cols = 6 - col;
        var rows = String.fromCharCode(this.EN_NUM[row].charCodeAt());
        s = cols + "_" + rows;
        //console.log(s);
        piece_blank = document.getElementById(cols + "_" + rows);
        return piece_blank;
    }
    /*
    if( this.name == "CC")
    {
      var div = this.new_div_shape("piece", 0, 0, this.blank_size, this.blank_size, this.piece_font_size);
    }
    else if( this.name == "CDC")
    {
      //var div = this.new_div_shape("piece", top, left, this.blank_size, this.blank_size, this.piece_font_size);  
      var div = this.new_div_shape("piece", -3, -3, this.blank_size, this.blank_size, this.piece_font_size);
    }
    var piece_str = this.new_div_shape(className, -2, -2, this.blank_size, this.blank_size, this.piece_font_size);
    piece_str.append(text);
    div.append(piece_str);
    var piece_blank = null;
    if( this.name == "CC" )
    {
      div.onclick = this.piece_click;
      piece_blank = document.getElementById(row + "_" + col);
      piece_blank.appendChild(div);
      return piece_blank;
    }
    else if( this.name == "CDC" )
    {
      var r = (9 - row);
      div.onclick = this.piece_click;
      var cols = String.fromCharCode(this.EN_NUM[col].charCodeAt() - 65248);
      var rows = String.fromCharCode(this.STR_NUM[r].charCodeAt() - 65248);
      piece_blank = document.getElementById(cols + "_" + rows);
      piece_blank.appendChild(div);
      //console.log("CDC:", this.EN_NUM[col] + "_" + this.STR_NUM[r]);
      return piece_blank;
    }*/
    return div;
  },
  new_bound: function (row, col, text) {
    var top = row * (this.blank_size + this.border_width);
    var left = col * (this.blank_size + this.border_width);
    var div = this.new_div_shape("boundary", top, left, this.blank_size, this.blank_size, this.bound_font_size);
    div.append(text);
    return div;
  },
  new_die_piece: function (row, col, className, text) {
    var top = row * (this.die_blank_size + this.border_width);
    var left = col * (this.die_blank_size + this.border_width);
    var div = this.new_div_shape("piece", top, left, this.die_blank_size, this.die_blank_size, this.die_piece_font_size);
    //console.log(className);
    var clr = "";
    var pie = "";
    for (k in this.PCE_MAP) {
      if (this.PCE_MAP[k].ch == text) {
        pie = this.PCE_MAP[k].en;
        clr = this.PCE_MAP[k].clr;
        break;
      }
    }
    var val = this.PCE_QUA.get(pie)
    div.id = "" + pie + val;
    //console.log(val, this.PCE_QUA);
    this.PCE_QUA.set(pie, (val - 1));
    if (className == "piece_dark") {
      clr = clr.toLowerCase();
      var piece_str = this.new_div_shape("piece_" + clr, -2, -2, this.die_blank_size, this.die_blank_size, this.die_piece_font_size);
      piece_str.append(text);
      div.append(piece_str);
      /* mouse click */
      div.onclick = this.piece_click;
      clr = null;
      pie = null;
      return div;
    }
    else if (className == "piece_dark_red" || className == "piece_dark_black") {
      var piece_str = this.new_div_shape(className, -2, -2, this.die_blank_size, this.die_blank_size, this.die_piece_font_size);
      piece_str.append(text);
      div.append(piece_str);
      return div;
    }
    else {
      var piece_str = this.new_div_shape("piece_dark", -2, -2, this.die_blank_size, this.die_blank_size, this.die_piece_font_size);
      piece_str.append(text);
      div.append(piece_str);
      piece_str.style.backgroundColor = "#FFFFFF";
      return div;
    }
  },
  new_movelist_title: function (text) {
    var height = this.movelist_font_size * 1.5;
    var width = (text.length + 2) * this.movelist_font_size;
    var left = 1 * (this.blank_size + this.border_width);
    var div = this.new_div_shape("label", 0, left, height, width, this.movelist_font_size);
    div.append(text);
    return div;
  },
  new_observation_movelist_title: function (text) {
    var height = this.movelist_font_size * 1.5;
    var width = (text.length + 2) * this.movelist_font_size;
    var left = 1.5 * (this.blank_size + this.border_width);
    var div = this.new_div_shape("label", 0, left, height, width, this.movelist_font_size);
    div.append(text);
    return div;
  },
  new_movelist: function (row, col) {
    var height = row * (this.blank_size + this.border_width);
    var width = col * (this.blank_size + this.border_width);
    var top = this.movelist_font_size * 2;
    return this.new_div_shape("path", top, 0, height, width, height);
  },
  new_movelist_move: function (row, col, type, text) {
    var height = this.movelist_font_size * 1.8;
    var width = (text.length + 1) * this.movelist_font_size;
    var top = row * (this.movelist_font_size) * 2;
    var left = col * (this.blank_size + this.border_width) + this.movelist_font_size * 2 + 3;
    var div_num = this.new_div_shape_line_height("move_num", top, 0, height, this.movelist_font_size * 2, this.movelist_font_size, this.movelist_font_size * 1.3);
    var div_move = this.new_div_shape_line_height(type, top, left, height, width, this.movelist_font_size, this.movelist_font_size * 1.3);
    var div = document.createElement("div");

    div.setAttribute("top", top);
    div_move.append(text);
    div_num.append((row + 1) + ".");
    div.append(div_num);
    div.append(div_move);
    if (this.mode != 1) {
      div.onclick = function () { board.history_move(parseInt(top / 24)) };
    }
    return div;
  },
  new_observation_movelist_move: function (row, col, type, text) {
    //console.log("new_observation_movelist_move: " 
    //  + board.player + " " + row + " " + col + " " + type + " " + text);
    var height = this.movelist_font_size * 1.8;
    var width = (text.length + 1) * this.movelist_font_size;
    var top = row * (this.movelist_font_size) * 2;

    if (type != "time1" && type != "time2") {
      var left = 0 * (this.blank_size + this.border_width) + this.movelist_font_size * 2 + 3;
      var div_num = this.new_div_shape_line_height("move_num", top, 0, height, this.movelist_font_size * 2, this.movelist_font_size, this.movelist_font_size * 1.3);
      var div_move = this.new_div_shape_line_height(type, top, left, height, width, this.movelist_font_size, this.movelist_font_size * 1.3);
    } else {

      var left = 2 * (this.blank_size + this.border_width) + this.movelist_font_size * 2 - 3;
      var div_move = this.new_div_shape_line_height(type, top, left, height, 8 * this.movelist_font_size, this.movelist_font_size, this.movelist_font_size * 1.3);
    }

    var div = document.createElement("div");

    div.setAttribute("top", top);
    div_move.append(text);
    if (type != "time1" && type != "time2") {
      div_num.append((row + 1) + ".");
      div.append(div_num);
    }
    div.onclick = function () { board.history_move(parseInt(top / 24)) };
    div.append(div_move);
    return div;
  },
  new_endgame_movelist_move: function (row, col, type, text, end_score, end_type) {
    let _this = this;
    var left = function (col) { return col * (_this.blank_size + _this.border_width) + _this.movelist_font_size * 2 + 3; }
    var height = this.movelist_font_size * 1.8;
    var width = (text.length + 1) * this.movelist_font_size;
    var top = row * (this.movelist_font_size) * 2;
    var div_num = this.new_div_shape_line_height("move_num", top, 0, height, this.movelist_font_size * 2, this.movelist_font_size, this.movelist_font_size * 1.3);
    var div_move = this.new_div_shape_line_height(type, top, left(col), height, width, this.movelist_font_size, this.movelist_font_size * 1.3);
    var div_scr = this.new_div_shape_line_height("end_score", top, left(col + 2), height, this.movelist_font_size * 2, this.movelist_font_size, this.movelist_font_size * 1.3);
    var div_type = this.new_div_shape_line_height("end_type", top, left(col + 2.6), height, this.movelist_font_size * 2, this.movelist_font_size, this.movelist_font_size * 1.3);
    var div = document.createElement("div");
    div.setAttribute("top", top);
    div_move.append(text);
    div_num.append(row + ".");
    div_scr.append(end_score);
    div_type.append(end_type);
    div.append(div_num);
    div.append(div_move);
    div.append(div_scr);
    div.append(div_type);
    return div;
  },
  new_right_arrow: function (row, col, height, width) {
    var top = row * (this.blank_size + this.border_width);
    var left = col * (this.blank_size + this.border_width);
    return this.new_div_shape("right_arrow", top, left, height, width, height);
  },
  new_left_arrow: function (row, col, height, width) {
    var top = row * (this.blank_size + this.border_width);
    var left = col * (this.blank_size + this.border_width);
    return this.new_div_shape("left_arrow", top, left, height, width, height);
  },
  new_number_move: function (row, col, height, width, text) {
    var top = row * (this.blank_size + this.border_width);
    var left = col * (this.blank_size + this.border_width);
    var div = this.new_div_shape("label", top, left, height, width, this.movelist_font_size);
    div.append(text);
    return div;
  },

  toSide: function (pce) {
    return this.PCE_MAP[pce].clr;
  },

  toStr: function (pce) {
    switch (this.LANG) {
      case this.LANG_EN:
        return this.PCE_MAP[pce].en;
      case this.LANG_CH:
        return this.PCE_MAP[pce].ch;
    }
  },

  toClassName: function (side) {
    return this.PCE_TYPE[side].className;
  },
  set_board: function (board) { }
}

BaseBoard.prototype.init = function () {
  //console.log("init: " + this.name, this.mode);
  var link = document.createElement('link');
  link.href = "chess.css";
  link.rel = "stylesheet";
  var tagN = document.getElementsByTagName("link");
  if (tagN.length == 0) {
    document.getElementsByTagName('head')[0].append(link);
  }
  this.screen_init();
  this.board_init();
  this.chess_board_init();
  if (this.name == "CC") {
    this.CC_board_init();
  }
  this.piece_init();
  this.red_died_init();
  if (this.name == "EC") {
    this.blue_died_init();
  } else {
    this.black_died_init();
  }

  this.div_movelist.append(this.new_observation_movelist_title("走步"));
  this.div_path = this.new_movelist(this.piece_row, 5);

  this.div_movelist.append(this.div_path);
  //console.log(typeof document.body, this.mode);

  document.body.onkeydown = this.keyEvent;
  document.body.onkeyup = this.metaKeyUp;

  //if( !this.mode ){add_input();}
  /*var bt = document.createElement("input");
  bt.id = "hisbtn";
  bt.type = "button";
  bt.value = "歷史走步";
  bt.onclick = function(){window.open('history.html', 'History', config='height=500,width=800');}
  var bget = this.div_endgame_movelist;
  bget.append(bt);*/
  //console.log("init end");
};

function his_init(index) {
  //console.log("his index", index, board.mode);
  if (board.mode == 1 && board.isstopboard == false) { return; }
  if (document.getElementById("set_mode") != null &&
    document.getElementById("set_mode").checked == false) {
    return;
  }
  let record_genmove = board.record_genmove;
  let turn = board.turn;
  let stop_flag = board.isstopboard;
  let openf = board.his_openf;
  let clr = board.clr;
  let buf = board.actlist;
  let tbuf = board.acttime;
  let data = board.board_data;
  let mode = board.mode;
  let info_r = board.info_row;
  let infoerr_r = board.info_error_row;
  let ply = board.player;
  var players = board.player_name;
  var sock = board.socket;
  let div_endgame = document.getElementById("div_endgame_movelist_CDC");
  let mui = document.getElementById("mui_sw");
  let mui1 = document.getElementById("mui_sw1");
  let f = true; let f1 = true;
  let highlight = board.highlight;
  let initBoard = board.initBoard;
  let initRemainDarkChess = board.initRemainDarkChess;

  if (mui != null) {
    f = mui.checked;
    f1 = mui1.checked;
  }

  board = null;
  document.body.innerHTML = "";
  document.body.setAttribute("overflow-x", "hidden");
  document.body.setAttribute("overflow-y", "hidden");
  board = new CDCBoard(mode);
  board.record_genmove = record_genmove;
  board.turn = turn;
  board.isstopboard = stop_flag;
  board.socket = sock;
  board.clr = clr;
  board.step = index - 1;
  board.his_step = index;
  board.actlist = buf;
  board.player = ply;
  board.his_openf = openf;
  board.player_name = players;
  board.highlight = highlight;
  board.initBoard = initBoard;
  board.initRemainDarkChess = initRemainDarkChess;

  //console.log("color", board.clr);
  // history
  if (board.mode == 0 || board.mode == 4) {
    let sw_lab = document.createElement("span");
    sw_lab.textContent = "使用者標準輸出自動換行:";
    sw_lab.style = "float: left;";

    let sw_lab1 = document.createElement("span");
    sw_lab1.textContent = "使用者標準錯誤輸出自動換行:";
    sw_lab1.style = "float: left;";

    let sw = document.createElement("input");
    sw.id = "mui_sw";
    sw.className = "mui-switch mui-switch-animbg";
    sw.type = "checkbox";
    sw.checked = f;

    let sw1 = document.createElement("input");
    sw1.id = "mui_sw1";
    sw1.className = "mui-switch mui-switch-animbg";
    sw1.type = "checkbox";
    sw1.checked = f1;

    sw_lab.appendChild(sw);
    sw_lab1.appendChild(sw1);
    board.div_endgame_movelist.insertBefore(sw_lab1, board.div_endgame_movelist.childNodes[0]);
    board.div_endgame_movelist.insertBefore(sw_lab, board.div_endgame_movelist.childNodes[0]);
    board.acttime = tbuf;
    board.board_data = data;
    board.info_row = info_r;
    board.info_error_row = infoerr_r;
    while (div_endgame.firstChild) {
      let clone = div_endgame.firstChild.cloneNode(false);
      board.div_endgame_movelist.appendChild(clone);
      //console.log(clone);
      div_endgame.removeChild(div_endgame.firstChild);
    }
    board.create_div_open_save();

    let sp = document.createElement("span");
    sp.append(openf);
    sp.style.float = "center";
    board.div_endgame_movelist.insertBefore(sp, board.div_endgame_movelist.childNodes[0]);

    window.document.title = openf;
  }
  // abort
  if (board.mode == 2) {
    let abort_clone = div_endgame.firstChild.cloneNode(true);
    board.div_endgame_movelist.appendChild(abort_clone);
  }
  //observation
  if (board.mode == 3 || board.isstopboard) {
    while (div_endgame.firstChild) {
      let _clone = div_endgame.firstChild.cloneNode(true);
      div_endgame.removeChild(div_endgame.firstChild);
      board.div_endgame_movelist.appendChild(_clone);
    }
    document.getElementById("set_mode").onchange = CBclick;
  }
  if (document.getElementById("message_button") != null) {
    document.getElementById("message_button").addEventListener("click", get_Chat_Content);
  }
  if (document.getElementById("message_input") != null) {
    document.getElementById("message_input").addEventListener("keyup", function (event) {
      // Number 13 is the "Enter" key on the keyboard
      if (event.keyCode === 13) {
        // Cancel the default action, if needed
        event.preventDefault();
        // Trigger the button element with a click
        document.getElementById("message_button").click();
      }
    });
  }

  let openChess = [0,0,0,0,0,0,0,0,0,0,0,0,0,0];
  let intitOpenChess = [1,2,2,2,2,2,5,1,2,2,2,2,2,5];
  let pce_array = ['-', 'K', 'G', 'M', 'R', 'N', 'C', 'P', 'X', 'k', 'g', 'm', 'r', 'n', 'c', 'p'];
  let pce_array2 = ['K', 'G', 'M', 'R', 'N', 'C', 'P', 'k', 'g', 'm', 'r', 'n', 'c', 'p'];
  for(let i = 0; i < board.initRemainDarkChess.length; i++){
    intitOpenChess[i] -= parseInt(board.initRemainDarkChess[i]);
  }
  console.log(board.initRemainDarkChess);
  console.log(board.initBoard);
  // make initboard
  for(let i = 0; i < initBoard.length; i++){
    let chessId = parseInt(initBoard[i]);
    let pos = String.fromCharCode('a'.charCodeAt(0) + (i % 4)) + '_' + ('8' - parseInt(i / 4));
    console.log(pos);
    if (chessId == 0) { // 無子
      _InitMakeBoard(1, pos, "");
    }
    if (chessId >= 1 && chessId <= 7) {
      _InitMakeBoard(0, pos, pce_array[chessId]);
      openChess[chessId - 1]++;
    }
    if (chessId >= 9 && chessId <= 15) {
      _InitMakeBoard(0, pos, pce_array[chessId]);
      openChess[chessId - 2]++;
    }
  }
  // make init piece state
  console.log(intitOpenChess);
  console.log(openChess);
  for(let i = 0; i < 14; i++){
    let diePiece = intitOpenChess[i] - openChess[i];
    for(let j = 0; j < diePiece; j++){
      console.log(pce_array2[i]);
      Die_pie(pce_array2[i], "flip", i > 6 ? "piece_red" : "piece_black");
      Die_pie(pce_array2[i], "move", "piece_dark");
    }
  }

  buf = null;
  tbuf = null;
  data = null;
  clr = null;
  info = null;
  var move = null;
  var move_pos = null;
  var latt = null;
  var ratt = null;
  for (var i = 0; i < board.actlist.length; i++) {
    var strl = board.actlist[i];
    let text = strl.split(" ");
    var num = (i == index && board.highlight) ? 2 : 0;
    strl = strl.split(' ')[1];
    console.log("board.highlight:", board.highlight);
    //console.log("now step:", i);
    board.div_path.append(
      move = board.new_observation_movelist_move(i, (i % 2 == 0) ? 0 : 2,
        "move" + ((board.clr + i) % 2 + 1 + num), text[1])
    );
    board.div_path.append(
      board.new_observation_movelist_move(i, (i % 2 == 0) ? 0 : 2,
        "time" + ((board.clr + i) % 2 + 1),
        ((parseInt(text[3]) / 1000).toFixed(1) + " / " + (parseInt(text[4]) / 1000).toFixed(1)))
    );

    if (i == (index - 1)) { latt = move.getAttribute("top"); }
    if (i == (index + 1)) { ratt = move.getAttribute("top"); }
    if (i == (index - 7)) { move_pos = move.getAttribute("top"); }
  }

  board.div_movelist.append(board.new_left_arrow(board.piece_row + 1, 0, 18, 30));
  board.div_movelist.append(board.new_right_arrow(board.piece_row + 1, 3, 18, 30));
  var la = board.div_movelist.getElementsByClassName("left_arrow")[0];
  var ra = board.div_movelist.getElementsByClassName("right_arrow")[0];

  board.div_movelist.append(board.new_number_move(board.piece_row + 1, 1,
    board.movelist_font_size * 1.8, board.movelist_font_size * 5,
    "" + (index + 1) + "/" + (board.actlist.length))
  );



  if (latt != null) {
    la.setAttribute("top", latt);
    la.onclick = function () {
      board.history_move(parseInt(latt / 24))
    };
  } else {
    //console.log(la, board.actlist.length - 1);
    la.setAttribute("top", board.actlist.length - 1);
    la.onclick = function () {
      board.history_move(parseInt(board.actlist.length - 1));
    };
  }
  if (ratt != null) {
    ra.setAttribute("top", ratt);
    ra.onclick = function () { board.history_move(parseInt(ratt / 24)) };
  } else {
    //console.log(ra, board.actlist.length - 1);
    ra.setAttribute("top", 0);
    ra.onclick = function () {
      board.history_move(parseInt(0));
    };
  }

  board.div_path.scrollTop = move_pos;
  la = null; ra = null;

  //history
  if (board.mode == 0 || board.mode == 4) {
    setPos(index + 1);
    var vs = new Array();
    if (board.board_data == "") {
      //console.log(board.board_data);
      vs[0] = "先手", vs[2] = "後手";
    } else {
      vs = board.board_data.split("* ")[2].split(" ");
    }
    let name_lab1 = document.createElement("span");
    name_lab1.textContent = vs[0];
    name_lab1.style.float = "left";
    name_lab1.style.backgroundColor = (board.clr == 0) ? "red" : "black";
    name_lab1.style.color = "white";
    name_lab1.style.fontSize = "18px";

    let lab_vs = document.createElement("span");
    lab_vs.textContent = "vs";
    lab_vs.style.paddingLeft = "10px";
    lab_vs.style.paddingRight = "10px";
    lab_vs.style.float = "left";
    lab_vs.style.color = "black";
    lab_vs.style.fontSize = "18px";


    let name_lab2 = document.createElement("span");
    name_lab2.textContent = vs[2];
    name_lab2.style.float = "left";
    name_lab2.style.backgroundColor = (board.clr == 1) ? "red" : "black";
    name_lab2.style.color = "white";
    name_lab2.style.fontSize = "18px";

    board.div_endgame_movelist.insertBefore(name_lab2, board.div_endgame_movelist.childNodes[0]);
    board.div_endgame_movelist.insertBefore(lab_vs, board.div_endgame_movelist.childNodes[0]);
    board.div_endgame_movelist.insertBefore(name_lab1, board.div_endgame_movelist.childNodes[0]);
  }
  if (board.mode == 2) {
    //console.log(div_endgame);
    let box = document.getElementById("box_dialog");
    let text_step = 1;
    var yn = "no";
    //console.log(box);
    if (box != null) {
      //console.log("box exist");
      let text = document.getElementById("abort_step");
      if (text != null) {
        text.value = index + 1;
        text_step = text.value;
      }
      let r = document.getElementById("radio_1");
      if (r != null) {
        if (r.checked == true) {
          yn = "yes";
        }
      }
      //console.log("悔棋", "雙方是否交換顏色?", text_step, yn, r.checked, r);
      box.remove();
      DEBUG_SET("悔棋", "雙方是否交換顏色?", text_step, yn);
    } else {
      //console.log("box null");
      text_step = document.getElementsByClassName("path")[0].childNodes.length;
      DEBUG_SET("悔棋", "雙方是否交換顏色?", text_step, yn);
    }
  }
  window = board;
}


var posarr = [0, 0];
// set textarea position
function setPos(ind) {
  //console.log("setPos", ind);

  //scrollHeight all
  //clientHeight see
  let sw = document.getElementById("mui_sw");
  let sw1 = document.getElementById("mui_sw1");
  let textArea = document.getElementById("user_info");
  let textArea1 = document.getElementById("info_error");
  if (sw.checked == true) {
    let lineHeight = textArea.scrollHeight / board.info_row[board.info_row.length - 2];
    textArea.scrollTop = lineHeight * board.info_row[ind];
    posarr[0] = textArea.scrollTop;
  } else {
    textArea.scrollTop = posarr[0];
  }
  if (sw1.checked == true) {
    let lineHeight1 = textArea1.scrollHeight / board.info_error_row[board.info_error_row.length - 2];
    textArea1.scrollTop = lineHeight1 * board.info_error_row[ind];
    posarr[1] = textArea1.scrollTop;
  } else {
    textArea1.scrollTop = posarr[1];
  }
}
function focus_textbox() {
  //console.log("textbox");
  board.his_create_name = true;
}
function blur_textbox() {
  board.his_create_name = false;
}
function _InitMakeBoard(action, pos, piece) {
  console.log(action, pos, piece);
  var getpos = document.getElementById(pos);
  if(action == 0){  // flip
    let t = board.PCE_MAP[piece].ch;
    if (getpos.hasChildNodes()) {
      getpos.firstElementChild.style.backgroundColor = "#a5a4a4";
      getpos.firstElementChild.firstElementChild.append(t);
    }
    if (piece.match(/([a-z])+/)) {
      getpos.firstElementChild.firstElementChild.className = "piece_black";
      Die_pie(piece, "flip", "piece_black");
    }
    else if (piece.match(/([A-Z])+/)) {
      getpos.firstElementChild.firstElementChild.className = "piece_red";
      Die_pie(piece, "flip", "piece_red");
    }
  }else{  // remove dark piece
    while (getpos.firstChild) {
      getpos.removeChild(getpos.firstChild);
    }
  }
}
function _Flip(getstr, piece, ply) {
  var bk = document.getElementById(getstr);
  if(board.clr == -1){
    board.clr = (board.PCE_MAP[board.actlist[0].split(" ")[1][3]].clr == "RED") ? 0 : 1;
  }
  let t = board.PCE_MAP[piece].ch;
  if (bk.hasChildNodes()) {
    bk.firstElementChild.style.backgroundColor = "#a5a4a4";
    bk.firstElementChild.firstElementChild.append(t);
  }
  if (piece.match(/([a-z])+/)) {
    bk.firstElementChild.firstElementChild.className = "piece_black";
    Die_pie(piece, "flip", "piece_black");
  }
  else if (piece.match(/([A-Z])+/)) {
    bk.firstElementChild.firstElementChild.className = "piece_red";
    Die_pie(piece, "flip", "piece_red");
  }
  board.tmp = null;

  var clr_str = "blue";

  if (board.clr == 0) {
    clr_str = (ply % 2 == 0) ? "red" : "black";
  } else {
    clr_str = (ply % 2 == 0) ? "black" : "red";
  }

  var div = document.createElement("div");
  div.className = clr_str + "_rect";
  div.style.cssText = "width: 38px";
  div.style.cssText += "height: 38px";
  div.style.cssText += "top: 0.7px";
  div.style.cssText += "left: 0.7px";
  bk.append(div);
  return;
}
function _Move(sta_p, tar_p, k) {
  //var stanum = ((arr[1][0].charCodeAt() - 'a'.charCodeAt()) * 8) 
  //+ (arr[1][1].charCodeAt() - '0'.charCodeAt());
  //var tarnum = ((arr[1][3].charCodeAt() - 'a'.charCodeAt()) * 8) 
  //+ (arr[1][4].charCodeAt() - '0'.charCodeAt());
  //var tarid = "" + arr[1][3] + "_" + arr[1][4];
  var tar = document.getElementById(tar_p);
  //var staid = "" + arr[1][0] + "_" + arr[1][1];
  var sta = document.getElementById(sta_p);
  //console.log(sta);
  var pie = sta.firstElementChild;
  var dpie = null;
  if (tar.hasChildNodes()) { dpie = tar.firstElementChild; }
  while (sta.firstChild) {
    sta.removeChild(sta.firstChild);
  }
  while (tar.firstChild) {
    tar.removeChild(tar.firstChild);
  }
  if(board.clr == -1){
    board.clr = (pie.firstChild.className == "piece_red") ? 0 : 1;
  }
  // arrow //
  var set_top = 0;
  var set_left = 0;
  var w = 0;
  var h = 0;
  //var k = stanum - tarnum;
  var arrow_class = "";
  switch (true) {
    case (k >= 1 && k <= 7):
      arrow_class = "d_arrow";
      w = 1;
      h = k;
      set_top = remove_px(sta.style.top) + 20;
      set_left = remove_px(sta.style.left);
      break;
    case (k >= -7 && k <= -1):
      arrow_class = "u_arrow";
      w = 1;
      h = Math.abs(k);
      set_top = remove_px(tar.style.top) + 20;
      set_left = remove_px(tar.style.left);
      break;
    case (k >= 8 && k <= 24):
      arrow_class = "l_arrow";
      w = (k / 8);
      h = 1;
      set_top = remove_px(tar.style.top);
      set_left = remove_px(tar.style.left) + 20;
      break;
    case (k >= -24 && k <= -8):
      arrow_class = "r_arrow";
      w = (Math.abs(k) / 8);
      h = 1;
      set_top = remove_px(sta.style.top);
      set_left = remove_px(sta.style.left) + 20;
      break;
  }
  // arrow end //
  var div = document.createElement("div");
  div.id = "arrow";
  div.className = arrow_class;
  div.style.cssText = "width: " + (w * 40) + "px";
  div.style.cssText += "height: " + (h * 40) + "px";
  div.style.cssText += "top: " + set_top + "px";
  div.style.cssText += "left: " + set_left + "px";
  var pie_board = document.getElementById("div_piece_CDC");
  pie_board.append(div);
  //console.log(div);
  div = null;
  div = document.createElement("div");
  div.className = "orange_rect";
  div.style.cssText = "width: 38px";
  div.style.cssText += "height: 38px";
  div.style.cssText += "top: 0.7px";
  div.style.cssText += "left: 0.7px";
  tar.append(div);
  //console.log(div);
  tar.appendChild(pie);
  if (dpie != null && dpie.hasChildNodes() == true) {
    var dtext = dpie.firstElementChild.textContent;
    var pieen = null;
    for (k in board.PCE_MAP) {
      if (board.PCE_MAP[k].ch == dtext) {
        pieen = board.PCE_MAP[k].en;
        Die_pie(pieen, "move", "piece_dark");
        break;
      }
    }
  }
  board.tmp = null;
  board.after_blank = null;
  board.remove_div("rect");
}
// create save div
BaseBoard.prototype.create_div_open_save = function () {
  let bget = board.div_endgame_movelist;
  bget.style.width = "50%";
  let div = document.createElement("div");
  var style = "width:100%;";
  style += "position: relative;";
  style += "float: left;";
  div.setAttribute("style", style);
  let fopen = document.createElement("input");
  fopen.id = "myfile";
  fopen.type = "file";
  fopen.accept = ".txt";
  fopen.onchange = Open;
  fopen.style = "float: left;";

  let fname = document.createElement("input");
  fname.id = "filename";
  fname.type = "text";
  fname.style = "float: left;";
  fname.onclick = focus_textbox;
  fname.onblur = blur_textbox;

  let btn = document.createElement("input");
  btn.id = "submit";
  btn.value = "save";
  btn.type = "button";
  btn.style.top = "400px";
  btn.onclick = Save;
  btn.style = "float: left;";
  if (board.mode == 0) {
    div.appendChild(fopen);
  }
  div.appendChild(fname);
  div.appendChild(btn);
  bget.appendChild(div);

  bget.insertBefore(div, bget.childNodes[0]);
}

// create information areatext
function add_input() {
  board.create_div_open_save();
  //console.log("add_input");
  let bget = board.div_endgame_movelist;
  let uinfo = document.createElement("textarea");
  uinfo.id = "user_info";

  style = "";
  style += "float: left;";
  style += "font-family: Courier New;";
  style += "position: relative;";
  uinfo.setAttribute("style", style);

  uinfo.style.width = "100%";
  uinfo.style.maxWidth = "100%";
  uinfo.style.resize = "none";
  uinfo.style.overflowX = "scroll";
  uinfo.style.overflowY = "scroll";
  uinfo.style.whiteSpace = "nowrap";
  uinfo.rows = "20";

  let uinfoerr = document.createElement("textarea");
  uinfoerr.id = "info_error";

  style = "";
  style += "float: left;";
  style += "font-family: Courier New;";
  style += "position: relative;";
  uinfoerr.setAttribute("style", style);


  uinfoerr.style.width = "100%";
  uinfoerr.style.maxWidth = "100%";
  uinfoerr.style.resize = "none";
  uinfoerr.style.overflowX = "scroll";
  uinfoerr.style.overflowY = "scroll";
  uinfoerr.style.whiteSpace = "nowrap";
  uinfoerr.rows = "20";

  bget.appendChild(uinfo);
  bget.appendChild(uinfoerr);
}
function remove_px(text) {
  var str = text.split('px');
  return parseInt(str[0]);
}
BaseBoard.prototype.keyEvent = function (event) {

  var key = event.keyCode || event.which;
  var keychar = String.fromCharCode(key);
  //console.log("keyEvent", key);
  //console.log(board.his_create_name, "key");
  if (board.mode == 1 && board.isstopboard == false) { return; }

  if (key == this.exampleKey) {
    this.metaChar = true;
  }
  if (key != this.exampleKey) {
    if (this.metaChar) {
      this.metaChar = false;
    } else {
      //console.log(key);
      switch (key) {
        case 65:
        case 37:
        case 87:
        case 38:
          if (parseInt(board.his_step - 1) >= 0) {
            board.history_move(parseInt(board.his_step - 1));
          } else {
            //max step
            board.history_move(parseInt(board.actlist.length - 1));
          }
          break;
        case 68:
        case 39:
        case 83:
        case 40:
          if (parseInt(board.his_step + 1) < board.actlist.length) {
            board.history_move(parseInt(board.his_step + 1));
          } else {
            //first step
            board.history_move(parseInt(0));
          }
          break;
      }
    }
  }
}
BaseBoard.prototype.metaKeyUp = function (event) {
  //console.log("metaKeyUp");
  var key = event.keyCode || event.which;
  if (key == this.exampleKey) {
    this.metaChar = false;
  }
}
function Die_pie(pie, act, type) {
  var val = null;
  var diep = null;
  switch (act) {
    case "flip":
      //console.log("flip");
      val = board.PCE_QUA.get(pie);
      val = val + 1;
      diep = document.getElementById(pie + val);
      diep.firstElementChild.className = type;
      board.PCE_QUA.set(pie, val);
      break;
    case "move":
      val = board.PCE_DIE.get(pie);
      val = val + 1;
      diep = document.getElementById(pie + val);
      diep.firstElementChild.className = type;
      board.PCE_DIE.set(pie, val);
      break;
  }
  return;
}

BaseBoard.prototype.history_move = function (item) {
  //var item = parseInt(this.getAttribute("top")) / 24 ;
  if (document.getElementById("set_mode") != null &&
    document.getElementById("set_mode").checked == false) {
    return;
  }
  //console.log("history_move", item); 
  his_init(item);
  var moveid = -1;
  while (moveid < item) {
    board.remove_div("blue_rect");
    board.remove_div("black_rect");
    board.remove_div("red_rect");
    board.remove_div("orange_rect");
    var arow = document.getElementById("arrow");
    if (arow != null) { arow.remove(); }
    moveid++;
    let list = board.actlist[moveid];
    let str = list.split(' ')[1];

    //console.log(str);

    switch (str[2]) {
      case "(":
        //remove_rect();
        let getstr = "" + str[0] + "_" + str[1];
        _Flip(getstr, str[3], moveid);
        break;
      case "-":
        //remove_rect();
        var stanum = ((str[0].charCodeAt() - 'a'.charCodeAt()) * 8)
          + (str[1].charCodeAt() - '0'.charCodeAt());
        var tarnum = ((str[3].charCodeAt() - 'a'.charCodeAt()) * 8)
          + (str[4].charCodeAt() - '0'.charCodeAt());
        var tarid = "" + str[3] + "_" + str[4];
        var staid = "" + str[0] + "_" + str[1];
        var k = stanum - tarnum;
        _Move(staid, tarid, k);
        break;
    }
  }
}
BaseBoard.prototype.board_init = function () {
  //console.log("BB board_init");
  for (var row = 1; row <= this.blank_row; row++) {
    for (var col = 1; col <= this.blank_col; col++) {
      this.div_board.append(this.new_board_blank(row, col));
    }
  }
}
/* push piece blank */
BaseBoard.prototype.CC_board_init = function () {
  for (var row = 1; row <= this.blank_row + 1; row++) {
    for (var col = 1; col <= this.blank_col + 1; col++) {
      this.div_piece.append(this.new_board_blank(row, col, "CC_blank"));
    }
  }
}

BaseBoard.prototype.path_init = function (move, start, end, ply, this_ply) {
}


BaseBoard.prototype.movelist_init = function (moves, ply, url_first, url_last) {
  this.div_movelist.append(this.new_movelist_title("走步", 1));
  var new_url = function (_ply) { return url_first + (_ply) + url_last; }
  this.div_path = this.new_movelist(this.piece_row, 4);
  var row = 0;
  var move = null;
  var move_pos = 0;
  var max_ply = moves.length;

  //console.log(this.PCE_MAP[moves[0][3]].clr);
  var clr_1, clr_2;
  if (this.name == "CDC") {
    clr_1 = (this.PCE_MAP[moves[0][3]].clr == "RED" ? 3 : 4);
    clr_2 = (this.PCE_MAP[moves[0][3]].clr == "RED" ? 1 : 2);
  }
  else if (this.name == "CC") {
    clr_1 = 3;
    clr_2 = 1;
  }
  var option = "";
  for (var i = 0; i < max_ply; i++) {
    let url = new_url(i + 1);
    this.div_path.append(move = this.new_movelist_move(i, i % 2, "move" + (i == (ply - 1) ? clr_1 + i % 2 : clr_2 + i % 2), moves[i]));
    if ((i == (ply - 1) ? clr_1 + i % 2 : clr_2 + i % 2) > 2) { this.player = i; option = moves[i]; }
    move.onclick = function () { location.href = url; };
    if (i == (ply - 7)) move_pos = move.getAttribute("top");//style.top.substring(0, move.style.top.length - 2);
  }
  if (this.name == "CC") {
    CCBoard.player = this.player % 2;
    if (CCBoard.player == 0) {
      if (option[2] == "平") {
        var ind = this.CH_NUM.indexOf(option[1]) + 1 + "" + "_" + "" + (10 - this.CH_NUM.indexOf(option[1]));
        //console.log(ind);
        var element = document.getElementById(ind);
        var div = document.createElement("div");
        div.className = "move_mark";
        div.style.backgroundColor = "blue";
        element.append(div);
        //console.log(element);
      }
      else if (option[2] == "進") {
      }
    }
    else if (CCBoard.player == 1) {
      if (option[2] == "平") {
        var ind = pos_x + "" + "_" + "" + this.STR_NUM.indexOf(option[1]);
        //console.log(ind);
        var element = document.getElementById(ind);
        var div = document.createElement("div");
        div.className = "move_mark";
        div.style.backgroundColor = "blue";
        element.append(div);
        //console.log(element);
      }
      else if (option[2] == "進") {
      }
    }
  }
  else if (this.name == "CDC") {
    var en_pos = String.fromCharCode(option.charCodeAt(0) + 65248);
    var nu_pos = String.fromCharCode(option.charCodeAt(1) + 65248);
    switch (option[2]) {
      case "(":
        var element = document.getElementById(en_pos + "_" + nu_pos);
        var div = document.createElement("div");
        div.className = "move_mark";
        div.style.backgroundColor = "pink";
        div.style.left = "34px";
        div.style.top = "34px";
        element.append(div);
        break;
      case "-":
        var element = document.getElementById(en_pos + "_" + nu_pos);
        var div = document.createElement("div");
        div.className = "move_mark";
        div.style.backgroundColor = "blue";
        div.style.left = "34px";
        div.style.top = "34px";
        element.append(div);
        en_pos = String.fromCharCode(option.charCodeAt(3) + 65248);
        nu_pos = String.fromCharCode(option.charCodeAt(4) + 65248);
        var element = document.getElementById(en_pos + "_" + nu_pos);
        var div = document.createElement("div");
        div.className = "move_mark";
        div.style.left = "34px";
        div.style.top = "34px";
        element.append(div);
        break;
    }
  }
  this.div_movelist.append(this.div_path);
  this.div_movelist.append(this.new_left_arrow(this.piece_row + 1, 0, 18, 30));
  this.div_movelist.append(this.new_right_arrow(this.piece_row + 1, 3, 18, 30));
  var str = ply.toString() + "/" + max_ply.toString();
  this.div_movelist.append(this.new_number_move(this.piece_row + 1, 1, this.movelist_font_size * 1.8, this.movelist_font_size * 5, str));
  this.div_movelist.getElementsByClassName("left_arrow")[0].onclick = function () { location.href = new_url(Math.max(0, ply - 1)); }
  this.div_movelist.getElementsByClassName("right_arrow")[0].onclick = function () { location.href = new_url(Math.min(max_ply, ply + 1)); }
  this.div_path.scrollTop = move_pos;
}


BaseBoard.prototype.endgame_movelist_init = function (best_score, moves, best_index, url) {
  //console.log(moves);
  this.div_endgame_movelist.append(this.new_movelist_title("殘局", 1));
  var new_url = function (HIS) { return url + HIS; }
  //this.div_path = this.new_movelist(this.piece_row, 4);
  var row = 0;
  var move = null;
  var move_pos = 0;
  var max_ply = moves.length;

  var div_path = this.new_movelist(this.piece_row, 4);

  for (var i = 0; i < moves.length; i++) {
    let url = new_url(moves[i]);
    let s = moves[i].split(" ");
    let class_name = "end_move" + (i == best_index ? "_best" : "");
    div_path.append(move = this.new_endgame_movelist_move(i, 0, class_name, s[0], s[1], s[2]));
    move.onclick = function () { location.href = new_url(s[3]); };
    //if (i == (ply - 7)) move_pos = move.getAttribute("top");//style.top.substring(0, move.style.top.length - 2);
  }
  this.div_endgame_movelist.append(div_path);
  this.div_endgame_movelist.append(this.new_left_arrow(this.piece_row + 1, 0, 18, 30));
  this.div_endgame_movelist.append(this.new_right_arrow(this.piece_row + 1, 3, 18, 30));
  var str = best_score;//ply.toString() + "/" + max_ply.toString();
  this.div_endgame_movelist.append(this.new_number_move(this.piece_row + 1, 1, this.movelist_font_size * 1.8, this.movelist_font_size * 5, str));
  div_path.scrollTop = move_pos; //this.div_path.scrollHeight;
  this.div_endgame_movelist.getElementsByClassName("left_arrow")[0].onclick = function () { location.href = new_url(moves[best_index].split(" ")[3]); }
  this.div_endgame_movelist.getElementsByClassName("right_arrow")[0].onclick = function () { location.href = new_url(moves[best_index].split(" ")[3]); }
  //console.log(move_pos);
}



BaseBoard.prototype.set_status = function (moves, ply, prev_url, next_url) {
}


function CDCBoard(mode) {
  BaseBoard.call(this, 8, 4, "CDC", mode);
  this.init();
}
CDCBoard.prototype = new BaseBoard();

CDCBoard.prototype.chess_board_init = function () {
  //console.log("chess_board_init");
  this.div_board.append(this.new_backslash(4, 3));
  this.div_board.append(this.new_backslash(5, 4));
  this.div_board.append(this.new_slash(4, 4));
  this.div_board.append(this.new_slash(5, 3));

  for (j = 1, i = 8; j <= 8; j++, i--)
    this.div_board.append(this.new_bound(j, 0, this.STR_NUM[i]));
  for (i = 1; i <= 4; i++)
    this.div_board.append(this.new_bound(9, i, this.EN_NUM[i]));
  //console.log("chess_board_init end");
}

CDCBoard.prototype.piece_init = function () {
  //console.log("piece_init");
  for (var row = 1; row <= this.piece_row; row++) {
    for (var col = 1; col <= this.piece_col; col++) {
      this.div_piece.append(this.new_board_piece(row, col, this.toClassName("DARK"), this.PCE_MAP["D"].ch));
    }
  }
  //console.log("piece_init end");
}

CDCBoard.prototype.screen_init = function () {
  //console.log("screen_init");
  var style = "";
  style = "";
  style += "top :" + (0 * (this.blank_size + this.border_width) + this.boundary_top) + "px;"
  style += "left:" + (3 * (this.blank_size + this.border_width) + this.boundary_left) + "px;"
  this.div_board.setAttribute("style", style);

  style = "";
  style += "top :" + (0 * (this.blank_size + this.border_width) + this.piece_bound_top + this.boundary_top) + "px;"
  style += "left:" + (3 * (this.blank_size + this.border_width) + this.piece_bound_left + this.boundary_left) + "px;"
  this.div_piece.setAttribute("style", style);

  style = "";
  style += "top :" + ((0 * (this.die_blank_size + this.border_width)) + (0.8 * (this.blank_size + this.border_width)) + this.boundary_top) + "px;"
  style += "left:" + (0.5 * (this.blank_size + this.border_width) + this.boundary_left) + "px;"
  this.div_red_died.setAttribute("style", style);

  style = "";
  style += "top :" + ((5 * (this.die_blank_size + this.border_width)) + (0.8 * (this.blank_size + this.border_width)) + this.boundary_top) + "px;"
  style += "left:" + (0.5 * (this.blank_size + this.border_width) + this.boundary_left) + "px;"
  this.div_black_died.setAttribute("style", style);

  style = "";
  style += "top :" + (0.5 * (this.blank_size + this.border_width) + this.boundary_top) + "px;"
  style += "left:" + (8.5 * (this.blank_size + this.border_width) + this.boundary_left) + "px;"
  this.div_movelist.setAttribute("style", style);

  style = "";
  style += "top :" + (0.5 * (this.blank_size + this.border_width) + this.boundary_top) + "px;"
  style += "left:" + (14.5 * (this.blank_size + this.border_width) + this.boundary_left) + "px;"
  this.div_endgame_movelist.setAttribute("style", style);
  //console.log("screen_init end");
}

CDCBoard.prototype.red_died_init = function () {
  //console.log("red_died_init");
  this.div_red_died.append(this.new_die_piece(5, 0, this.PCE_TYPE["DARK_RED"].className, this.PCE_MAP["K"].ch));
  this.div_red_died.append(this.new_die_piece(0, 0, this.PCE_TYPE["DARK_RED"].className, this.PCE_MAP["P"].ch));
  this.div_red_died.append(this.new_die_piece(1, 0, this.PCE_TYPE["DARK_RED"].className, this.PCE_MAP["P"].ch));
  this.div_red_died.append(this.new_die_piece(2, 0, this.PCE_TYPE["DARK_RED"].className, this.PCE_MAP["P"].ch));
  this.div_red_died.append(this.new_die_piece(3, 0, this.PCE_TYPE["DARK_RED"].className, this.PCE_MAP["P"].ch));
  this.div_red_died.append(this.new_die_piece(4, 0, this.PCE_TYPE["DARK_RED"].className, this.PCE_MAP["P"].ch));
  this.div_red_died.append(this.new_die_piece(0, 1, this.PCE_TYPE["DARK_RED"].className, this.PCE_MAP["G"].ch));
  this.div_red_died.append(this.new_die_piece(0, 2, this.PCE_TYPE["DARK_RED"].className, this.PCE_MAP["G"].ch));
  this.div_red_died.append(this.new_die_piece(1, 1, this.PCE_TYPE["DARK_RED"].className, this.PCE_MAP["M"].ch));
  this.div_red_died.append(this.new_die_piece(1, 2, this.PCE_TYPE["DARK_RED"].className, this.PCE_MAP["M"].ch));
  this.div_red_died.append(this.new_die_piece(2, 1, this.PCE_TYPE["DARK_RED"].className, this.PCE_MAP["R"].ch));
  this.div_red_died.append(this.new_die_piece(2, 2, this.PCE_TYPE["DARK_RED"].className, this.PCE_MAP["R"].ch));
  this.div_red_died.append(this.new_die_piece(3, 1, this.PCE_TYPE["DARK_RED"].className, this.PCE_MAP["N"].ch));
  this.div_red_died.append(this.new_die_piece(3, 2, this.PCE_TYPE["DARK_RED"].className, this.PCE_MAP["N"].ch));
  this.div_red_died.append(this.new_die_piece(4, 1, this.PCE_TYPE["DARK_RED"].className, this.PCE_MAP["C"].ch));
  this.div_red_died.append(this.new_die_piece(4, 2, this.PCE_TYPE["DARK_RED"].className, this.PCE_MAP["C"].ch));
  //console.log("red_died_init end");
}

CDCBoard.prototype.black_died_init = function () {
  //console.log("black_died_init");
  this.div_black_died.append(this.new_die_piece(0, 2, this.PCE_TYPE["DARK_BLACK"].className, this.PCE_MAP["k"].ch));
  this.div_black_died.append(this.new_die_piece(1, 0, this.PCE_TYPE["DARK_BLACK"].className, this.PCE_MAP["p"].ch));
  this.div_black_died.append(this.new_die_piece(2, 0, this.PCE_TYPE["DARK_BLACK"].className, this.PCE_MAP["p"].ch));
  this.div_black_died.append(this.new_die_piece(3, 0, this.PCE_TYPE["DARK_BLACK"].className, this.PCE_MAP["p"].ch));
  this.div_black_died.append(this.new_die_piece(4, 0, this.PCE_TYPE["DARK_BLACK"].className, this.PCE_MAP["p"].ch));
  this.div_black_died.append(this.new_die_piece(5, 0, this.PCE_TYPE["DARK_BLACK"].className, this.PCE_MAP["p"].ch));
  this.div_black_died.append(this.new_die_piece(1, 1, this.PCE_TYPE["DARK_BLACK"].className, this.PCE_MAP["g"].ch));
  this.div_black_died.append(this.new_die_piece(1, 2, this.PCE_TYPE["DARK_BLACK"].className, this.PCE_MAP["g"].ch));
  this.div_black_died.append(this.new_die_piece(2, 1, this.PCE_TYPE["DARK_BLACK"].className, this.PCE_MAP["m"].ch));
  this.div_black_died.append(this.new_die_piece(2, 2, this.PCE_TYPE["DARK_BLACK"].className, this.PCE_MAP["m"].ch));
  this.div_black_died.append(this.new_die_piece(3, 1, this.PCE_TYPE["DARK_BLACK"].className, this.PCE_MAP["r"].ch));
  this.div_black_died.append(this.new_die_piece(3, 2, this.PCE_TYPE["DARK_BLACK"].className, this.PCE_MAP["r"].ch));
  this.div_black_died.append(this.new_die_piece(4, 1, this.PCE_TYPE["DARK_BLACK"].className, this.PCE_MAP["n"].ch));
  this.div_black_died.append(this.new_die_piece(4, 2, this.PCE_TYPE["DARK_BLACK"].className, this.PCE_MAP["n"].ch));
  this.div_black_died.append(this.new_die_piece(5, 1, this.PCE_TYPE["DARK_BLACK"].className, this.PCE_MAP["c"].ch));
  this.div_black_died.append(this.new_die_piece(5, 2, this.PCE_TYPE["DARK_BLACK"].className, this.PCE_MAP["c"].ch));
  //console.log("black_died_init end");
}

CDCBoard.prototype.set_status = function (status) {
  //console.log(status);
  for (var p in status) {
    var piece = p;
    var side = this.toSide(p);
    var open = status[p].open;
    var dark = status[p].dark;
    var died = status[p].died;
    var type = "";
    var n = open + dark;
    var row_s = 0,
      row_e = 0,
      row = 0;
    var col_s = 0,
      col_e = 0,
      col = 0;

    if (piece == "K") {
      row_s = 5;
      row_e = 5;
      col_s = 0;
      col_e = 0;
    }
    if (piece == "k") {
      row_s = -1;
      row_e = -1;
      col_s = 2;
      col_e = 2;
    }
    if (piece == "G" || piece == "g") {
      row_s = 0;
      row_e = 0;
      col_s = 1;
      col_e = 2;
    }
    if (piece == "M" || piece == "m") {
      row_s = 1;
      row_e = 1;
      col_s = 1;
      col_e = 2;
    }
    if (piece == "R" || piece == "r") {
      row_s = 2;
      row_e = 2;
      col_s = 1;
      col_e = 2;
    }
    if (piece == "N" || piece == "n") {
      row_s = 3;
      row_e = 3;
      col_s = 1;
      col_e = 2;
    }
    if (piece == "C" || piece == "c") {
      row_s = 4;
      row_e = 4;
      col_s = 1;
      col_e = 2;
    }
    if (piece == "P" || piece == "p") {
      row_s = 0;
      row_e = 4;
      col_s = 0;
      col_e = 0;
    }
    row = row_s;
    col = col_s;
    for (var i = 0; i < n; i++) {
      if (dark > 0) {
        type = "DARK_" + side;
        dark--;
      } else if (died > 0) {
        type = side;
        died--;
        open--;
      } else {
        type = "DARK";
        died--;
      }
      if (side == "RED")
        this.div_red_died.append(this.new_die_piece(row, col, this.PCE_TYPE[type].className, this.toStr(piece)));
      else
        this.div_black_died.append(this.new_die_piece(row + 1, col, this.PCE_TYPE[type].className, this.toStr(piece)));
      col++;
      if (col > col_e) {
        col = col_s;
        row++;
      }
    }
  }
}

CDCBoard.prototype.set_board = function (board) {
  //console.log("set_board");
  for (var i = 0; i < board.length; ++i) {
    if (board[i] == "E") continue;
    var side = this.toSide(board[i]);
    var str = this.toStr(board[i]);
    this.div_piece.append(this.new_board_piece(8 - (i % 8), 1 + parseInt(i / 8), this.toClassName(side), str));
  }
  //console.log("set_board end");
}


function CCBoard() {
  //console.log("CCBoard");
  BaseBoard.call(this, 9, 8, "CC");
  this.init();
  //console.log("call end");
  this.blank_row = 9;
  this.blank_col = 8;
  this.piece_bound_left = -(this.blank_size / 2)
  this.piece_bound_top = -(this.blank_size / 2)
}
CCBoard.prototype = new BaseBoard();
CCBoard.prototype.chess_board_init = function () {

  this.div_board.append(this.new_backslash(1, 4));
  this.div_board.append(this.new_backslash(2, 5));
  this.div_board.append(this.new_slash(1, 5));
  this.div_board.append(this.new_slash(2, 4));

  this.div_board.append(this.new_backslash(8, 4));
  this.div_board.append(this.new_backslash(9, 5));
  this.div_board.append(this.new_slash(8, 5));
  this.div_board.append(this.new_slash(9, 4));

  this.div_board.append(this.new_river(5, 1));

  for (i = 1; i <= 9; i++)
    this.div_board.append(this.new_bound(0 - 0.5, i - 0.5, this.STR_NUM[i]));
  for (i = 1, j = 9; i <= 9; i++, j--)
    this.div_board.append(this.new_bound(11 - 0.5, i - 0.5, this.CH_NUM[j]));
}

CCBoard.prototype.piece_init = function () {

  this.div_piece.append(this.new_board_piece(1, 5, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["k"].ch));
  this.div_piece.append(this.new_board_piece(1, 4, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["g"].ch));
  this.div_piece.append(this.new_board_piece(1, 6, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["g"].ch));
  this.div_piece.append(this.new_board_piece(1, 3, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["m"].ch));
  this.div_piece.append(this.new_board_piece(1, 7, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["m"].ch));
  this.div_piece.append(this.new_board_piece(1, 1, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["r"].ch));
  this.div_piece.append(this.new_board_piece(1, 9, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["r"].ch));
  this.div_piece.append(this.new_board_piece(1, 2, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["n"].ch));
  this.div_piece.append(this.new_board_piece(1, 8, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["n"].ch));
  this.div_piece.append(this.new_board_piece(3, 2, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["c"].ch));
  this.div_piece.append(this.new_board_piece(3, 8, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["c"].ch));
  this.div_piece.append(this.new_board_piece(4, 1, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["p"].ch));
  this.div_piece.append(this.new_board_piece(4, 3, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["p"].ch));
  this.div_piece.append(this.new_board_piece(4, 5, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["p"].ch));
  this.div_piece.append(this.new_board_piece(4, 7, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["p"].ch));
  this.div_piece.append(this.new_board_piece(4, 9, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["p"].ch));


  this.div_piece.append(this.new_board_piece(10, 5, this.PCE_TYPE["RED"].className, this.PCE_MAP["K"].ch));
  this.div_piece.append(this.new_board_piece(10, 4, this.PCE_TYPE["RED"].className, this.PCE_MAP["G"].ch));
  this.div_piece.append(this.new_board_piece(10, 6, this.PCE_TYPE["RED"].className, this.PCE_MAP["G"].ch));
  this.div_piece.append(this.new_board_piece(10, 3, this.PCE_TYPE["RED"].className, this.PCE_MAP["M"].ch));
  this.div_piece.append(this.new_board_piece(10, 7, this.PCE_TYPE["RED"].className, this.PCE_MAP["M"].ch));
  this.div_piece.append(this.new_board_piece(10, 1, this.PCE_TYPE["RED"].className, this.PCE_MAP["R"].ch));
  this.div_piece.append(this.new_board_piece(10, 9, this.PCE_TYPE["RED"].className, this.PCE_MAP["R"].ch));
  this.div_piece.append(this.new_board_piece(10, 2, this.PCE_TYPE["RED"].className, this.PCE_MAP["N"].ch));
  this.div_piece.append(this.new_board_piece(10, 8, this.PCE_TYPE["RED"].className, this.PCE_MAP["N"].ch));
  this.div_piece.append(this.new_board_piece(8, 2, this.PCE_TYPE["RED"].className, this.PCE_MAP["C"].ch));
  this.div_piece.append(this.new_board_piece(8, 8, this.PCE_TYPE["RED"].className, this.PCE_MAP["C"].ch));
  this.div_piece.append(this.new_board_piece(7, 1, this.PCE_TYPE["RED"].className, this.PCE_MAP["P"].ch));
  this.div_piece.append(this.new_board_piece(7, 3, this.PCE_TYPE["RED"].className, this.PCE_MAP["P"].ch));
  this.div_piece.append(this.new_board_piece(7, 5, this.PCE_TYPE["RED"].className, this.PCE_MAP["P"].ch));
  this.div_piece.append(this.new_board_piece(7, 7, this.PCE_TYPE["RED"].className, this.PCE_MAP["P"].ch));
  this.div_piece.append(this.new_board_piece(7, 9, this.PCE_TYPE["RED"].className, this.PCE_MAP["P"].ch));

}

CCBoard.prototype.screen_init = function () {
  var style = "";
  style = "";
  style += "top :" + (1 * (this.blank_size + this.border_width) + this.boundary_top) + "px;"
  style += "left:" + (3 * (this.blank_size + this.border_width) + this.boundary_left) + "px;"
  this.div_board.setAttribute("style", style);

  style = "";
  style += "top :" + (1 * (this.blank_size + this.border_width) + this.piece_bound_top + this.boundary_top) + "px;"
  style += "left:" + (3 * (this.blank_size + this.border_width) + this.piece_bound_left + this.boundary_left) + "px;"
  this.div_piece.setAttribute("style", style);

  style = "";
  style += "top :" + ((0 * (this.die_blank_size + this.border_width)) + (2.2 * (this.blank_size + this.border_width)) + this.boundary_top) + "px;"
  style += "left:" + (0.5 * (this.blank_size + this.border_width) + this.boundary_left) + "px;"
  //this.div_red_died.setAttribute("style", style);
  this.div_black_died.setAttribute("style", style);

  style = "";
  style += "top :" + ((6 * (this.die_blank_size + this.border_width)) + (2.2 * (this.blank_size + this.border_width)) + this.boundary_top) + "px;"
  style += "left:" + (0.5 * (this.blank_size + this.border_width) + this.boundary_left) + "px;"
  this.div_red_died.setAttribute("style", style);
  //this.div_black_died.setAttribute("style", style);

  style = "";
  style += "top :" + (1 * (this.blank_size + this.border_width) + this.boundary_top) + "px;"
  style += "left:" + (13 * (this.blank_size + this.border_width) + this.boundary_left) + "px;"
  this.div_movelist.setAttribute("style", style);
}

CCBoard.prototype.red_died_init = function () {
  for (var row = 0; row < 5; row++) {
    for (var col = 0; col < 3; col++)
      this.div_red_died.append(this.new_die_piece(row, col, this.toClassName("DARK"), this.PCE_MAP["E"].ch));
  }
  /*
    this.div_red_died.append(this.new_die_piece(0, 0, this.PCE_TYPE["RED"].className, this.PCE_MAP["P"].ch));
    this.div_red_died.append(this.new_die_piece(1, 0, this.PCE_TYPE["RED"].className, this.PCE_MAP["P"].ch));
    this.div_red_died.append(this.new_die_piece(2, 0, this.PCE_TYPE["RED"].className, this.PCE_MAP["P"].ch));
    this.div_red_died.append(this.new_die_piece(3, 0, this.PCE_TYPE["RED"].className, this.PCE_MAP["P"].ch));
    this.div_red_died.append(this.new_die_piece(4, 0, this.PCE_TYPE["RED"].className, this.PCE_MAP["P"].ch));
    this.div_red_died.append(this.new_die_piece(0, 1, this.PCE_TYPE["RED"].className, this.PCE_MAP["G"].ch));
    this.div_red_died.append(this.new_die_piece(0, 2, this.PCE_TYPE["RED"].className, this.PCE_MAP["G"].ch));
    this.div_red_died.append(this.new_die_piece(1, 1, this.PCE_TYPE["RED"].className, this.PCE_MAP["M"].ch));
    this.div_red_died.append(this.new_die_piece(1, 2, this.PCE_TYPE["RED"].className, this.PCE_MAP["M"].ch));
    this.div_red_died.append(this.new_die_piece(2, 1, this.PCE_TYPE["RED"].className, this.PCE_MAP["R"].ch));
    this.div_red_died.append(this.new_die_piece(2, 2, this.PCE_TYPE["RED"].className, this.PCE_MAP["R"].ch));
    this.div_red_died.append(this.new_die_piece(3, 1, this.PCE_TYPE["RED"].className, this.PCE_MAP["N"].ch));
    this.div_red_died.append(this.new_die_piece(3, 2, this.PCE_TYPE["RED"].className, this.PCE_MAP["N"].ch));
    this.div_red_died.append(this.new_die_piece(4, 1, this.PCE_TYPE["RED"].className, this.PCE_MAP["C"].ch));
    this.div_red_died.append(this.new_die_piece(4, 2, this.PCE_TYPE["RED"].className, this.PCE_MAP["C"].ch));
    */
}

CCBoard.prototype.black_died_init = function () {

  for (var row = 0; row < 5; row++) {
    for (var col = 0; col < 3; col++)
      this.div_black_died.append(this.new_die_piece(row, col, this.toClassName("DARK"), this.PCE_MAP["E"].ch));
  }
  /*
    this.div_black_died.append(this.new_die_piece(0, 0, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["p"].ch));
    this.div_black_died.append(this.new_die_piece(1, 0, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["p"].ch));
    this.div_black_died.append(this.new_die_piece(2, 0, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["p"].ch));
    this.div_black_died.append(this.new_die_piece(3, 0, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["p"].ch));
    this.div_black_died.append(this.new_die_piece(4, 0, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["p"].ch));
    this.div_black_died.append(this.new_die_piece(0, 1, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["g"].ch));
    this.div_black_died.append(this.new_die_piece(0, 2, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["g"].ch));
    this.div_black_died.append(this.new_die_piece(1, 1, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["m"].ch));
    this.div_black_died.append(this.new_die_piece(1, 2, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["m"].ch));
    this.div_black_died.append(this.new_die_piece(2, 1, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["r"].ch));
    this.div_black_died.append(this.new_die_piece(2, 2, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["r"].ch));
    this.div_black_died.append(this.new_die_piece(3, 1, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["n"].ch));
    this.div_black_died.append(this.new_die_piece(3, 2, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["n"].ch));
    this.div_black_died.append(this.new_die_piece(4, 1, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["c"].ch));
    this.div_black_died.append(this.new_die_piece(4, 2, this.PCE_TYPE["BLACK"].className, this.PCE_MAP["c"].ch));
    */
}

CCBoard.prototype.set_board = function (board) {
  //console.log(board);
  for (var i = 1; i <= 10; ++i) {
    for (var j = 1; j <= 9; ++j) {
      var idx = i * 10 + j;
      if (board[idx] == "E") continue;
      var side = this.toSide(board[idx]);
      var str = this.toStr(board[idx]);
      var class_name = this.toClassName(side);
      this.div_piece.append(this.new_board_piece(i, j, class_name, str));
    }
  }
}

CCBoard.prototype.set_status = function (status) {
  //console.log("status",status);
  for (var p in status) {
    var piece = p;
    var side = this.toSide(p);
    var open = status[p].open;
    var dark = status[p].dark;
    var died = status[p].died;
    var type = "";
    var n = open + dark;
    var row_s = 0,
      row_e = 0,
      row = 0;
    var col_s = 0,
      col_e = 0,
      col = 0;
    if (piece == "G" || piece == "g") {
      row_s = 0;
      row_e = 0;
      col_s = 1;
      col_e = 2;
    }
    if (piece == "M" || piece == "m") {
      row_s = 1;
      row_e = 1;
      col_s = 1;
      col_e = 2;
    }
    if (piece == "R" || piece == "r") {
      row_s = 2;
      row_e = 2;
      col_s = 1;
      col_e = 2;
    }
    if (piece == "N" || piece == "n") {
      row_s = 3;
      row_e = 3;
      col_s = 1;
      col_e = 2;
    }
    if (piece == "C" || piece == "c") {
      row_s = 4;
      row_e = 4;
      col_s = 1;
      col_e = 2;
    }
    if (piece == "P" || piece == "p") {
      row_s = 0;
      row_e = 4;
      col_s = 0;
      col_e = 0;
    }
    row = row_s;
    col = col_s;
    for (var i = 0; i < n; i++) {
      if (dark > 0) {
        type = "DARK_" + side;
        dark--;
      } else if (died > 0) {
        type = side;
        died--;
        open--;
      } else {
        type = "DARK";
        died--;
      }
      if (side == "RED")
        this.div_red_died.append(this.new_die_piece(row, col, this.PCE_TYPE[type].className, this.toStr(piece)));
      else
        this.div_black_died.append(this.new_die_piece(row, col, this.PCE_TYPE[type].className, this.toStr(piece)));
      col++;
      if (col > col_e) {
        col = col_s;
        row++;
      }
    }
  }
}
/* mouse event */
var add_value = [-1, 1];
var push_in = -1;
var push_s = "";
var S_NUM = ["０", "１", "２", "３", "４", "５", "６", "７", "８"];
var E_NUM = ["", "ａ", "ｂ", "ｃ", "ｄ"];
var S_NUM_url = ["0", "1", "2", "3", "4", "5", "6", "7", "8"];
var E_NUM_url = ["", "a", "b", "c", "d"];
var score = [
  ["將", ["帥", "仕", "相", "俥", "傌", "炮", ""]],
  ["士", ["仕", "相", "俥", "傌", "炮", "兵", ""]],
  ["象", ["相", "俥", "傌", "炮", "兵", ""]],
  ["車", ["俥", "傌", "炮", "兵", ""]],
  ["馬", ["傌", "炮", "兵", ""]],
  ["包", ["帥", "仕", "相", "俥", "傌", "炮", "兵", ""]],
  ["卒", ["帥", "兵", ""]],
  ["帥", ["將", "士", "象", "車", "馬", "包", ""]],
  ["仕", ["士", "象", "車", "馬", "包", "卒", ""]],
  ["相", ["象", "車", "馬", "包", "卒", ""]],
  ["俥", ["車", "馬", "包", "卒", ""]],
  ["傌", ["馬", "包", "卒", ""]],
  ["炮", ["將", "士", "象", "車", "馬", "包", "卒", ""]],
  ["兵", ["將", "卒"]]
];
var red_set = new Set(["帥", "仕", "相", "俥", "傌", "炮", "兵"]);
var black_set = new Set(["將", "士", "象", "車", "馬", "包", "卒"]);
var PCE_POS = new Map();

var pie_tmp = null;
var after_blank = null;
var Score_map = set_score();
function set_score() {
  var s_map = new Map(score);
  //.log(s_map);
  return s_map;
}
function create_div(w, h, t, l) {
  var div = document.createElement("div");
  div.className = "rect";
  //div.className = "move_mark";
  div.style.cssText = "width: " + w + "px";
  div.style.cssText += "height: " + h + "px";
  div.style.cssText += "top: " + t + "px";
  div.style.cssText += "left: " + l + "px";
  //console.log("create_div end", w, h, t, l);
  return div;
}
function is_barrier(id, text) {
  var opp = document.getElementById(id);
  //console.log("is barrier id:",id);
  if (opp.firstElementChild != null) {
    var oppt = opp.firstElementChild.textContent;
    if (opp.firstElementChild.className == "rect") {/*console.log("blank");*/return 0; }
    if (opp.firstElementChild.className == "piece"
      && opp.firstElementChild.firstElementChild.className == "piece_dark") {
      /*console.log("dark");*/
      return 1;
    }
    if (opp.firstElementChild.className == "piece" && Score_map.has(text)) {
      //console.log(Score_map.get(text).indexOf(chit));
      if (Score_map.get(text).indexOf(oppt) != -1) {/*console.log("opponent");*/return -1; }
      else {/*console.log("same");*/return 1; }
    }
  }
  //console.log("blank ", id);
  return 0;
}
function move_position(x, y, pos, text) {
  //console.log("move_position");
  var i;
  var flag = -1
  var flag_M = -1;
  var flag_N = -1;
  var barrier_pos = "";
  switch (text) {
    case "帥":
      if (x < 8 || x > 10 || y < 4 || y > 6) { break; }
      for (i = 0; i < add_value.length; i++) {
        if ((x + add_value[i]) >= 8 && (x + add_value[i]) <= 10) {
          push_s = (x + add_value[i]) + "_" + y;
          flag = is_barrier(push_s, text);
          if (flag == 1) {
            var div = create_div(43, 43, 0.5, 0.5);
            document.getElementById(push_s).appendChild(div);
          }
        }
        if ((y + add_value[i]) >= 4 && (y + add_value[i]) <= 6) {
          push_s = x + "_" + (y + add_value[i]);
          flag = is_barrier(push_s, text);
          if (flag == 1) {
            var div = create_div(43, 43, 0.5, 0.5);
            document.getElementById(push_s).appendChild(div);
          }
        }
      }
      break;
    case "將":
      if (x < 1 || x > 3 || y < 4 || y > 6) { break; }
      for (i = 0; i < add_value.length; i++) {
        if ((x + add_value[i]) >= 1 && (x + add_value[i]) <= 3) {
          push_s = (x + add_value[i]) + "_" + y;
          flag = is_barrier(push_s, text);
          if (flag == 1) {
            var div = create_div(43, 43, 0.5, 0.5);
            document.getElementById(push_s).appendChild(div);
          }
        }
        if ((y + add_value[i]) >= 4 && (y + add_value[i]) <= 6) {
          push_s = x + "_" + (y + add_value[i]);
          flag = is_barrier(push_s, text);
          if (flag == 1) {
            var div = create_div(43, 43, 0.5, 0.5);
            document.getElementById(push_s).appendChild(div);
          }
        }
      }
      break;
    case "仕":
      if (((x == 8 || x == 10) && y == 4) || ((x == 8 || x == 10) && y == 6) || (x == 9 && y == 5)) {
        for (i = 0; i < add_value.length; i++) {
          for (j = 0; j < add_value.length; j++) {
            if (((x + add_value[i]) >= 8 && ((x + add_value[i]) <= 10)) && ((y + add_value[j]) >= 4 && ((y + add_value[j]) <= 6))) {
              push_s = (x + add_value[i]) + "_" + (y + add_value[j])
              flag = is_barrier(push_s, text);
              if (flag == 1) {
                var div = create_div(43, 43, 0.5, 0.5);
                document.getElementById(push_s).appendChild(div);
              }
            }
          }
        }
      }
      break;
    case "士":
      if (((x == 1 || x == 3) && y == 4) || ((x == 1 || x == 3) && y == 6) || (x == 2 && y == 5)) {
        for (i = 0; i < add_value.length; i++) {
          for (j = 0; j < add_value.length; j++) {
            if (((x + add_value[i]) >= 1 && ((x + add_value[i]) <= 3)) && ((y + add_value[j]) >= 4 && ((y + add_value[j]) <= 6))) {
              push_s = (x + add_value[i]) + "_" + (y + add_value[j])
              flag = is_barrier(push_s, text);
              if (flag == 1) {
                var div = create_div(43, 43, 0.5, 0.5);
                document.getElementById(push_s).appendChild(div);
              }
            }
          }
        }
      }
      break;
    case "相":
      if (x >= 6 && x <= 10) {
        for (i = 0; i < add_value.length; i++) {
          for (j = 0; j < add_value.length; j++) {
            if (((x + add_value[i] + add_value[i]) >= 6 && ((x + add_value[i] + add_value[i]) <= 10))
              && ((y + add_value[j] + add_value[j]) >= 1 && ((y + add_value[j] + add_value[j]) <= 9))) {
              push_s = (x + add_value[i] + add_value[i]) + "_" + (y + add_value[j] + add_value[j])
              barrier_pos = (x + add_value[i]) + "_" + (y + add_value[j]);
              flag_M = is_barrier(barrier_pos, text);
              flag = is_barrier(push_s, text);
              if (flag == 1 && flag_M == 0) {
                var div = create_div(43, 43, 0.5, 0.5);
                document.getElementById(push_s).appendChild(div);
              }
            }
          }
        }
      }
      break;
    case "象":
      if (x >= 1 && x <= 5) {
        for (i = 0; i < add_value.length; i++) {
          for (j = 0; j < add_value.length; j++) {
            if (((x + add_value[i] + add_value[i]) >= 1 && ((x + add_value[i] + add_value[i]) <= 5))
              && ((y + add_value[j] + add_value[j]) >= 1 && ((y + add_value[j] + add_value[j]) <= 9))) {
              push_s = (x + add_value[i] + add_value[i]) + "_" + (y + add_value[j] + add_value[j])
              barrier_pos = (x + add_value[i]) + "_" + (y + add_value[j]);
              flag_M = is_barrier(barrier_pos, text);
              flag = is_barrier(push_s, text);
              if (flag == 1 && flag_M == 0) {
                var div = create_div(43, 43, 0.5, 0.5);
                document.getElementById(push_s).appendChild(div);
              }
            }
          }
        }
      }
      break;
    case "傌":
    case "馬":
      for (i = 0; i < add_value.length; i++) {
        if ((x + add_value[i] + add_value[i]) >= 1 && (x + add_value[i] + add_value[i]) <= 10) {
          barrier_pos = (x + add_value[i]) + "_" + y;
          flag_N = is_barrier(barrier_pos, text);
          if (flag_N == 0) {
            for (j = 0; j < add_value.length; j++) {
              if (y + add_value[j] >= 1 && y + add_value[j] <= 9) {
                push_s = (x + add_value[i] + add_value[i]) + "_" + (y + add_value[j]);
                flag = is_barrier(push_s, text);
                if (flag == 1) {
                  var div = create_div(43, 43, 0.5, 0.5);
                  document.getElementById(push_s).appendChild(div);
                }
              }
            }
          }
        }
        if ((y + add_value[i] + add_value[i]) >= 1 && (y + add_value[i] + add_value[i]) <= 9) {
          barrier_pos = x + "_" + (y + add_value[i]);
          flag_N = is_barrier(barrier_pos, text);
          if (flag_N == 0) {
            for (j = 0; j < add_value.length; j++) {
              if (x + add_value[j] >= 1 && x + add_value[j] <= 9) {
                push_s = (x + add_value[j]) + "_" + (y + add_value[i] + add_value[i]);
                flag = is_barrier(push_s, text);
                if (flag == 1) {
                  var div = create_div(43, 43, 0.5, 0.5);
                  document.getElementById(push_s).appendChild(div);
                }
              }
            }
          }
        }
      }
      break;
    case "俥":
    case "車":
      for (i = 1; i < 10; i++) {
        if ((x - i) > 0) {
          push_s = (x - i) + "_" + y;
          flag = is_barrier(push_s, text);
          if (flag == 1) { break; }
          var div = create_div(43, 43, 0.5, 0.5);
          document.getElementById(push_s).appendChild(div);
          if (flag == -1) { break; }
        }
      }
      for (i = 1; i < 10; i++) {
        if ((x + i) <= 10) {
          push_s = (x + i) + "_" + y;
          flag = is_barrier(push_s, text);
          if (flag == 1) { break; }
          var div = create_div(43, 43, 0.5, 0.5);
          document.getElementById(push_s).appendChild(div);
          if (flag == -1) { break; }
        }
      }
      for (i = 1; i < 10; i++) {
        if ((y - i) > 0) {
          push_s = x + "_" + (y - i);
          flag = is_barrier(push_s, text);
          if (flag == 1) { break; }
          var div = create_div(43, 43, 0.5, 0.5);
          document.getElementById(push_s).appendChild(div);
          if (flag == -1) { break; }
        }
      }
      for (i = 1; i < 10; i++) {
        if ((y + i) < 10) {
          push_s = x + "_" + (y + i);
          flag = is_barrier(push_s, text);
          if (flag == 1) { break; }
          var div = create_div(43, 43, 0.5, 0.5);
          document.getElementById(push_s).appendChild(div);
          if (flag == -1) { break; }
        }
      }
      break;
    case "炮":
    case "包":
      var flag_C = -1;
      for (i = 1; i < 10; i++) {
        if ((x - i) > 0) {
          push_s = (x - i) + "_" + y;
          flag = is_barrier(push_s, text);
          if (flag == 1 || flag == -1) { flag_C = 1; break; }
          var div = create_div(43, 43, 0.5, 0.5);
          document.getElementById(push_s).appendChild(div);
        }
      }
      if (flag_C == 1) {
        for (i = i + 1; i < 10; i++) {
          if ((x - i) > 0) {
            push_s = (x - i) + "_" + y;
            flag = is_barrier(push_s, text);
            if (flag == 1) { break; }
            if (flag == -1) {
              var div = create_div(43, 43, 0.5, 0.5);
              document.getElementById(push_s).appendChild(div);
              break;
            }
          }
        }
        flag_C = 0;
      }
      for (i = 1; i < 10; i++) {
        if ((x + i) <= 10) {
          push_s = (x + i) + "_" + y;
          flag = is_barrier(push_s, text);
          if (flag == 1 || flag == -1) { flag_C = 1; break; }
          var div = create_div(43, 43, 0.5, 0.5);
          document.getElementById(push_s).appendChild(div);
        }
      }
      if (flag_C == 1) {
        for (i = i + 1; i < 10; i++) {
          if ((x + i) <= 10) {
            push_s = (x + i) + "_" + y;
            flag = is_barrier(push_s, text);
            if (flag == 1) { break; }
            if (flag == -1) {
              var div = create_div(43, 43, 0.5, 0.5);
              document.getElementById(push_s).appendChild(div);
              break;
            }
          }
        }
        flag_C = 0;
      }
      for (i = 1; i < 10; i++) {
        if ((y - i) > 0) {
          push_s = x + "_" + (y - i);
          flag = is_barrier(push_s, text);
          if (flag == 1 || flag == -1) { flag_C = 1; break; }
          var div = create_div(43, 43, 0.5, 0.5);
          document.getElementById(push_s).appendChild(div);
        }
      }
      if (flag_C == 1) {
        for (i = i + 1; i < 10; i++) {
          if ((y - i) > 0) {
            push_s = x + "_" + (y - i);
            flag = is_barrier(push_s, text);
            if (flag == 1) { break; }
            if (flag == -1) {
              var div = create_div(43, 43, 0.5, 0.5);
              document.getElementById(push_s).appendChild(div);
              break;
            }
          }
        }
        flag_C = 0;
      }
      for (i = 1; i < 10; i++) {
        if ((y + i) < 10) {
          push_s = x + "_" + (y + i);
          flag = is_barrier(push_s, text);
          if (flag == 1 || flag == -1) { flag_C = 1; break; }
          var div = create_div(43, 43, 0.5, 0.5);
          document.getElementById(push_s).appendChild(div);
        }
      }
      if (flag_C == 1) {
        for (i = i + 1; i < 10; i++) {
          if ((y + i) < 10) {
            push_s = x + "_" + (y + i);
            flag = is_barrier(push_s, text);
            if (flag == 1) { break; }
            if (flag == -1) {
              var div = create_div(43, 43, 0.5, 0.5);
              document.getElementById(push_s).appendChild(div);
              break;
            }
          }
        }
        flag_C = 0;
      }
      break;
    case "兵":
      if (x == 6 || x == 7) {
        push_s = (x - 1) + "_" + y;
        flag = is_barrier(push_s, text);
        if (flag == 1) { break; }
        var div = create_div(43, 43, 0.5, 0.5);
        document.getElementById(push_s).appendChild(div);
      }
      else if (x >= 2 && x <= 5) {
        push_s = (x - 1) + "_" + y;
        flag = is_barrier(push_s, text);
        if (flag != 1) {
          var div = create_div(43, 43, 0.5, 0.5);
          document.getElementById(push_s).appendChild(div);
        }
        for (i = 0; i < add_value.length; i++) {
          if ((y + add_value[i]) >= 1 && (y + add_value[i]) <= 9) {
            push_s = x + "_" + (y + add_value[i]);
            flag = is_barrier(push_s, text);
            if (flag != 1) {
              var div = create_div(43, 43, 0.5, 0.5);
              document.getElementById(push_s).appendChild(div);
            }
          }
        }
      }
      else if (x == 1) {
        for (i = 0; i < add_value.length; i++) {
          if ((y + add_value[i]) >= 1 && (y + add_value[i]) <= 9) {
            push_s = x + "_" + (y + add_value[i]);
            flag = is_barrier(push_s, text);
            if (flag != 1) {
              var div = create_div(43, 43, 0.5, 0.5);
              document.getElementById(push_s).appendChild(div);
            }
          }
        }
      }
      break;
    case "卒":
      if (x == 4 || x == 5) {
        push_s = (x + 1) + "_" + y;
        flag = is_barrier(push_s, text);
        if (flag == 1) { break; }
        var div = create_div(43, 43, 0.5, 0.5);
        document.getElementById(push_s).appendChild(div);
      }
      else if (x >= 6 && x <= 9) {
        push_s = (x + 1) + "_" + y;
        flag = is_barrier(push_s, text);
        if (flag != 1) {
          var div = create_div(43, 43, 0.5, 0.5);
          document.getElementById(push_s).appendChild(div);
        }
        for (i = 0; i < add_value.length; i++) {
          if ((y + add_value[i]) >= 1 && (y + add_value[i]) <= 9) {

            push_s = x + "_" + (y + add_value[i]);
            flag = is_barrier(push_s, text);
            if (flag != 1) {
              var div = create_div(43, 43, 0.5, 0.5);
              document.getElementById(push_s).appendChild(div);
            }
          }
        }
      }
      else if (x == 10) {
        for (i = 0; i < add_value.length; i++) {
          if ((y + add_value[i]) >= 1 && (y + add_value[i]) <= 9) {

            push_s = x + "_" + (y + add_value[i]);
            flag = is_barrier(push_s, text);
            if (flag != 1) {
              var div = create_div(43, 43, 0.5, 0.5);
              document.getElementById(push_s).appendChild(div);
            }
          }
        }
      }
      break;
  }

}
BaseBoard.prototype.remove_div = function (name) {
  var element = document.getElementsByClassName(name);
  while (element.length > 0) { element[0].remove(); }
}
CCBoard.prototype.piece_click = function () {
  //console.log("CC piece click", CCBoard.player );
  board.remove_div("rect");
  if (this.firstElementChild.className == "piece_red" && board.player == 1) {
    if (board.tmp != null) {
      board.tmp.style.backgroundColor = "#a5a4a4";
    }
    this.style.backgroundColor = "#60d386";
    after_blank = this.parentNode;
    var s = String(after_blank.id);
    var index = s.split('_');
    var x = parseInt(index[0]);
    var y = parseInt(index[1]);
    move_position(x, y, 10, this.firstElementChild.textContent);
    board.tmp = this;
  }
  else if (this.firstElementChild.className == "piece_black" && board.player == 0) {
    if (board.tmp != null) {
      board.tmp.style.backgroundColor = "#a5a4a4";
    }
    this.style.backgroundColor = "#60d386";
    after_blank = this.parentNode;
    var s = String(after_blank.id);
    var index = s.split('_');
    var x = parseInt(index[0]);
    var y = parseInt(index[1]);
    move_position(x, y, 10, this.firstElementChild.textContent);
    board.tmp = this;
  }
};
CCBoard.prototype.shape_mouse_click = function () {
  //console.log("CC shape_mouse_click");
  if (board.tmp != null && this.firstElementChild != board.tmp) {
    board.tmp.style.backgroundColor = "#a5a4a4";
    if (this.childNodes.length > 0) {
      var b_children = this.childNodes;
      var is_rect = b_children[b_children.length - 1];
      if (is_rect.className == "rect") {
        while (b_children.length > 0) {
          is_rect = b_children[b_children.length - 1];
          is_rect.remove();
        }
        this.appendChild(board.tmp);
        //console.log(this.id);
      }
      board.tmp = null;
      after_blank = null;
    }
    board.remove_div("rect");
  }
}
CDCBoard.prototype.piece_click = function () {
  //console.log("CDC click ", board.player, board.tmp);
  //console.log("piece_click", this);
  remove_rect();

  if (board.turn == 0 || board.mode != 1) { return; }

  if (this.firstElementChild.className.split('_')[1] == "red" && (board.clr + board.player) % 2 == 1) {
    return;
  }
  else if (this.firstElementChild.className.split('_')[1] == "black" && (board.clr + board.player) % 2 == 0) {
    return;
  }

  if (board.tmp != null) {
    board.tmp.style.backgroundColor = "#a5a4a4";
  } else {
    board.tmp = this;
    board.tmp.style.backgroundColor = "#a5a4a4";
  }

  var pathnode = (board.div_path.childNodes.length > 0) ? true : false;
  //console.log("board.clr", board.clr);
  var canm = false;
  if (pathnode) {
    //console.log(board.div_path.lastChild.lastChild.className);
    let path_node = board.div_path.childNodes[board.div_path.childNodes.length - 1];
    //red :move1, move3
    //console.log("board.div_path：",board.div_path.childNodes[board.div_path.childNodes.length-2] );
    if (path_node.lastChild.className == "time1") { canm = true; }
    if (path_node.lastChild.className == "move1") { canm = true; }
    if (path_node.lastChild.className == "move3") { canm = true; }
    //black :move2, move4
    if (path_node.lastChild.className == "time2") { canm = false; }
    if (path_node.lastChild.className == "move2") { canm = false; }
    if (path_node.lastChild.className == "move4") { canm = false; }

  }

  //console.log(canm, "---------",this.firstElementChild.className, board.clr, board.tmp);
  switch (this.firstElementChild.className) {
    case "piece_dark_red":
    case "piece_dark_black":
      after_blank = this.parentNode;
      this.style.backgroundColor = "#60d386";
      board.tmp.style.height = "40px";
      board.tmp.style.width = "40px";
      board.tmp.onmousedown = onclick_right(event, this);
      break;
    case "piece_dark":
      board.remove_div("rect");
      this.style.backgroundColor = "#60d386";
      //CDCBoard.tmp.style.height = "40px";
      //CDCBoard.tmp.style.width = "40px";
      board.tmp = this;
      break;
    case "piece_red":
      var childNode = this.parentNode.childNodes;
      //var canm = (board.clr % 2) == (board.player % 2)? true: false;
      //var canm = (board.clr == 0) ? true: false;
      if (!canm && childNode.length < 2) {
        board.remove_div("rect");
        this.style.backgroundColor = "#60d386";
        board.tmp = this;
      }
      if (canm && childNode.length < 2) { board.remove_div("rect"); }
      break;
    case "piece_black":
      var childNode = this.parentNode.childNodes;
      //var canm = (board.clr % 2 == 1) == (board.player % 2 == 0)? true: false;
      //var canm = (board.clr == 1) ? true: false;
      if (canm && childNode.length < 2) {
        board.remove_div("rect");
        this.style.backgroundColor = "#60d386";
        board.tmp = this;
      }
      if (!canm && childNode.length < 2) { board.remove_div("rect"); }
      break;
  }
}
CDCBoard.prototype.shape_mouse_click = function () {
  //console.log("shape mouse click ", board.turn, board.clr, this, board.tmp);
  if (board.mode != 1 || board.turn == 0) { return; }
  if (board.tmp == null) {
    if (this.firstElementChild.firstElementChild.className.split('_')[1] == "red" && board.clr == 1) {
      return;
    }
    else if (this.firstElementChild.firstElementChild.className.split('_')[1] == "black" && board.clr == 0) {
      return;
    }
  }
  if (this.childNodes.length < 1) {
    remove_rect();
    board.remove_div("rect");
  }
  if (board.turn == 1 && board.tmp != null
    && this.firstElementChild != board.tmp) {
    //console.log("1", board.tmp.textContent);
    board.tmp.style.backgroundColor = "#a5a4a4";
    if (this.childNodes.length > 0) {
      var b_children = this.childNodes;
      var is_rect = b_children[b_children.length - 1];
      if (is_rect.className == "rect") {
        var items = after_blank.id.split("_");
        var str1 = items.join("");
        items = this.id.split("_");
        var str2 = items.join("");
        var sendstr = "move " + str1 + "-" + str2;
        board.socket.send(sendstr);
        remove_rect();
        console.log(sendstr);
        board.turn = 0;
        board.record_genmove = "";
      }else{
        board.remove_div("rect");
      }
    }
  }
  else if (board.turn == 1 && board.tmp != null
    && board.tmp.firstElementChild.className == "piece_dark") {
    //console.log("2", board.tmp.textContent);
    after_blank = this;
    if (this.firstElementChild != null) {
      if (this.firstElementChild.className == "piece") {
        var items = this.id.split("_");
        var str = items.join("");
        var sendstr = "flip " + str + "-" + str;
        //console.log("send " + sendstr, board.socket);
        board.socket.send(sendstr);
        remove_rect();
        console.log(sendstr);
        board.turn = 0;
        board.record_genmove = "";
      }
    }
  }
  else if (board.turn == 1 && board.tmp != null && this.firstElementChild == board.tmp) {
    //console.log("3", board.tmp.textContent);
    var s = String(this.id);
    var index = s.split('_');
    var tmp_x = index[0];
    var tmp_y = index[1];
    var ind_x = E_NUM_url.indexOf(tmp_x);
    var ind_y = S_NUM_url.indexOf(tmp_y);
    var text = board.tmp.textContent;
    if (this.firstElementChild.firstElementChild.className == "rect") {
      //console.log("4", "rect", board.tmp.context);
      if (this.childNodes.length > 0) {
        var b_children = this.childNodes;
        var is_rect = b_children[b_children.length - 1];
        if (is_rect.className == "rect") {
          /*while(b_children.length > 0)
          {
            is_rect = b_children[b_children.length-1];
            is_rect.remove();
          }*/
          //move
          //this.appendChild(board.tmp);
          var items = after_blank.id.split("_");
          var str1 = items.join("");
          items = this.id.split("_");
          var str2 = items.join("");
          var sendstr = "move " + str1 + "-" + str2;
          context.log("send " + sendstr);
          board.socket.send(sendstr);
          remove_rect();
          console.log(sendstr);
          board.turn = 0;
          board.record_genmove = "";
        }
        //board.tmp = null;
        //after_blank = null;
      }
    }
    else {
      //console.log(Score_map.get(text));
      board.remove_div("rect");
      for (var i = 0; i < add_value.length; i++) {
        //console.log("i ", i);
        if ((ind_x + add_value[i]) > 0 && (ind_x + add_value[i]) < E_NUM_url.length) {
          var push_s = E_NUM_url[ind_x + add_value[i]] + "_" + S_NUM_url[ind_y];
          var pie_text = document.getElementById(push_s).textContent;
          var ok = is_barrier(push_s, text);
          //console.log( s, "x ", E_NUM_url[ind_x + add_value[i]], ":", ok,);

          if (text == "炮" || text == "包") {
            //console.log(push_s);
            if (ok == 0) {
              var div = create_div(38, 38, 0.7, 0.7);
              document.getElementById(push_s).appendChild(div);
            }
            move_C(ind_x, ind_y, text);
          }
          else if ((text != "炮" && text != "包") && ok != 1) {
            var div = create_div(38, 38, 0.7, 0.7);
            document.getElementById(push_s).appendChild(div);
          }
        }
        if ((ind_y + add_value[i]) > 0 && (ind_y + add_value[i]) < S_NUM_url.length) {
          var push_s = E_NUM_url[ind_x] + "_" + S_NUM_url[ind_y + add_value[i]];
          var pie_text = document.getElementById(push_s).textContent;
          var ok = is_barrier(push_s, text);
          //console.log( s, "y ", S_NUM_url[ind_y + add_value[i]], ":", ok,);
          if (text == "炮" || text == "包") {
            //console.log(push_s);
            if (ok == 0) {
              var div = create_div(38, 38, 0.7, 0.7);
              document.getElementById(push_s).appendChild(div);
            }
            move_C(ind_x, ind_y, text);
          }
          else if ((text != "炮" && text != "包") && ok != 1) {
            var div = create_div(38, 38, 0.7, 0.7);
            document.getElementById(push_s).appendChild(div);
          }

        }
      }
    }
    after_blank = this;
  }
}
CDCBoard.prototype.onclick_right = function (event) {
  document.oncontextmenu = function () { window.event.returnValue = false; }
  //console.log("onclick_right");
  if (event.button == 2) {
    if (this.firstElementChild.className == "piece_red" || this.firstElementChild.className == "piece_black") {
      this.firstElementChild.className = "piece_dark";
      var p = PCE_POS.get(pie).firstElementChild;
      var context = p.textContent;
      if (red_set.has(context)) { p.className = "piece_dark_red"; }
      else { p.className = "piece_dark_black"; }
    }
    else if (this.firstElementChild.className == "piece_dark") {
      var context = this.firstElementChild.textContent;
      var p = PCE_POS.get(pie).firstElementChild;
      if (red_set.has(context)) { this.firstElementChild.className = "piece_red"; p.className = "piece_dark"; }
      else { this.firstElementChild.className = "piece_black"; p.className = "piece_dark"; }
    }
  }
}

function move_C(x, y, text) {
  var flag_C = -1;
  var i = 0;
  for (i = 1; i < 4; i++) {
    if ((x - i) > 0) {
      var push_s = E_NUM_url[x - i] + "_" + S_NUM_url[y];
      var flag = is_barrier(push_s, text);
      if (flag == 1 || flag == -1) { flag_C = 1; break; }
    }
  }
  if (flag_C == 1) {
    for (var j = i + 1; j < 4; j++) {
      if ((x - j) > 0) {
        var push_s = E_NUM_url[x - j] + "_" + S_NUM_url[y];
        var flag = is_barrier(push_s, text);
        //var bbb = document.getElementById(push_s).firstElementChild;
        if (flag == 1) { break; }
        if (flag == -1) {
          //console.log(push_s, i, j);
          var div = create_div(38, 38, 0.7, 0.7);
          document.getElementById(push_s).appendChild(div);
          break;
        }
      }
    }
  }
  flag_C = 0;
  for (i = 1; i < 4; i++) {
    if ((x + i) <= 4) {
      var push_s = E_NUM_url[x + i] + "_" + S_NUM_url[y];
      var flag = is_barrier(push_s, text);
      if (flag == 1 || flag == -1) { flag_C = 1; break; }
    }
  }
  if (flag_C == 1) {
    for (var j = i + 1; j < 4; j++) {
      if ((x + j) <= 4) {
        var push_s = E_NUM_url[x + j] + "_" + S_NUM_url[y];
        var flag = is_barrier(push_s, text);
        //var bbb = document.getElementById(push_s).firstElementChild;
        if (flag == 1) { break; }
        if (flag == -1) {
          //console.log(push_s, i , j);
          var div = create_div(38, 38, 0.7, 0.7);
          document.getElementById(push_s).appendChild(div);
          break;
        }
      }
    }
  }
  flag_C = 0;
  for (i = 1; i < 8; i++) {
    if ((y - i) > 0) {
      var push_s = E_NUM_url[x] + "_" + S_NUM_url[y - i];
      var flag = is_barrier(push_s, text);
      if (flag == 1 || flag == -1) { flag_C = 1; break; }
    }
  }
  if (flag_C == 1) {
    for (var j = i + 1; j < 8; j++) {
      if ((y - j) > 0) {
        var push_s = E_NUM_url[x] + "_" + S_NUM_url[y - j];
        var flag = is_barrier(push_s, text);
        //var bbb = document.getElementById(push_s).firstElementChild;
        if (flag == 1) { break; }
        if (flag == -1) {
          //console.log(push_s, i, j);
          var div = create_div(38, 38, 0.7, 0.7);
          document.getElementById(push_s).appendChild(div);
          break;
        }
      }
    }
  }
  flag_C = 0;
  for (i = 1; i < 8; i++) {
    if ((y + i) <= 8) {
      var push_s = E_NUM_url[x] + "_" + S_NUM_url[y + i];
      var flag = is_barrier(push_s, text);
      if (flag == 1 || flag == -1) { flag_C = 1; break; }
    }
  }
  if (flag_C == 1) {
    for (var j = i + 1; j < 8; j++) {
      if ((y + j) <= 8) {
        var push_s = E_NUM_url[x] + "_" + S_NUM_url[y + j];
        var flag = is_barrier(push_s, text);
        //var bbb = document.getElementById(push_s).firstElementChild;
        if (flag == 1) { break; }
        if (flag == -1) {
          //console.log(push_s, i ,j);
          var div = create_div(38, 38, 0.7, 0.7);
          document.getElementById(push_s).appendChild(div);
          break;
        }
      }
    }
    flag_C = 0;
  }
  //console.log("move_C return");
  return;
}
function remove_rect() {
  board.remove_div("blue_rect");
  board.remove_div("black_rect");
  board.remove_div("red_rect");
  board.remove_div("orange_rect");
  var arow = document.getElementById("arrow");
  if (arow != null) { arow.remove(); }
}
function OS() {
  board.os_type = (navigator.platform.indexOf("Win", 0) != -1) ? '\r\n' : '\n';
}
function Save() {
  if (board.his_openf == "") { return; }
  var satext = "";
  var datacr = new Array();
  datacr = board.board_data.split('* ');
  //console.log(datacr);
  //"This is TEMPORARILY file. You can use it for DEBUG.";
  //20190721
  if (/\n/.test(datacr[0]) == false) {
    //satext += "This is TEMPORARILY file. You can use it for DEBUG." + board.os_type;
  }

  satext += "* " + datacr[1] + board.os_type;
  satext += "* " + datacr[2] + board.os_type;
  satext += "* " + datacr[3] + board.os_type;
  //console.log(satext);
  satext += "* " + datacr[4] + board.os_type;
  var i;
  for (i = 5; i <= 11; i++) {
    satext += "* " + datacr[i] + board.os_type;
  }
  //satext += board.os_type;

  satext += "* " + datacr[12] + board.os_type;
  satext += "* " + datacr[13] + board.os_type;
  //actlist
  console.log(board.actlist, board.his_step);
  var index = 1;
  var flag = false;
  for (i = 0; i <= board.his_step; i++) {
    if ((i % 2) == 0) {
      satext += "* " + index + ". ";
      index++;
    }
    satext += board.actlist[i].split(' ')[1];
    if ((i % 2) == 0) {
      satext += " ";
      flag = true;
    }else{
      satext += board.os_type;
      flag = false;
    }
  }
  if (flag) {
    satext += board.os_type;
  }
  satext += "* Comment 0 0";
  satext += board.os_type;

  board.acttime[0] = board.acttime[0].replace(/\r?\n/g, "");
  satext += "0. " + board.acttime[0].split(board.split_str)[0].split(' ')[1] + board.os_type;
  index = 1;
  for (var i = 1; i < (board.his_step + 2); i++) {
    board.acttime[i] = board.acttime[i].replace(/\r?\n/g, "");
    satext += index + ". " + board.acttime[i].split(board.split_str)[0].split(' ')[1] + board.os_type;
    index++;
  }
  satext += "# #" + board.os_type;
  satext += "# #" + board.os_type;
  satext += "###" + board.os_type;
  var txtblob = new Blob([satext], { type: "text/plain;charset=utf-8" });
  var textarea = document.getElementById("filename").value;
  var filename = "";

  if (textarea != "") {
    filename = textarea + ".txt";
  }
  else {
    filename = "board.txt";
  }
  let url = window.URL.createObjectURL(txtblob);

  let downloadNode = document.createElement("a");
  downloadNode.style.display = "none";
  downloadNode.href = url;
  downloadNode.download = filename;

  document.body.appendChild(downloadNode);
  downloadNode.click();
  window.URL.revokeObjectURL(url);
}



//=============================               MS               =============================//

function MSBoard() {
  BaseBoard.call(this, 5, 5, "MS");
  //this.border_width
  this.image_size = 80;
  this.blank_size = 82.3;
  //this.border_width = ;
  //20190601/
  //this.img_date = "20190601/"
  //20191129/
  this.img_date = "20191129/";
  this.img_dir = "MSimg";

  this.clr = 0; // 0: fir, 1: sec

  this.PCE = ["King", "Jade", "Gold", "Silver", "Bishop", "Rook", "Pawn"];

  this.STR_NUM = ["0", "1", "2", "3", "4", "5"];
  this.EN_NUM = ["", "a", "b", "c", "d", "e"];
  this.chose_pos = null;
  this.chose_FS = null;
  this.init();
}


MSBoard.prototype = new BaseBoard;

MSBoard.prototype.chess_board_init = function () {
  var div = document.createElement('div');
  var style = "";
  style = "";
  style += "top :" + (this.blank_size + this.border_width - 28) + "px;";
  style += "left:" + (this.blank_size + this.border_width - 16) + "px;";
  div.setAttribute("style", style);


  let img = new Image(480, 480);
  img.src = './' + this.img_dir + '/' + this.img_date + 'MSBoard.png';
  div.append(img);
  this.div_board.appendChild(div);
}

MSBoard.prototype.piece_init = function () {
  //console.log("piece_init");

  /* 5 4 3 2 1
   *------------
   *|r b s g k |a
   *|        p |b
   *|          |c
   *|P         |d
   *|K G S B R |e
   *------------
   * FEN=>rbsgk/4p/5/p4/KGSBR,
   * / => \n,
   * number => space,
   * r=>Rook, b=>Bishop, s=>silver, g=>gold, k=>king, p=>pawn
   */

  for (let j = 2; j < 6; j++) {
    let i = 1;
    var top = i * (this.blank_size + this.border_width);
    var left = (6 - j) * (this.blank_size + this.border_width);
    var div = this.get_blank(i, (6 - j));

    var style = "";
    style = "";
    style += "top :" + top + "px;";
    style += "left:" + left + "px;";
    div.setAttribute("style", style);

    let img = new Image(this.image_size, this.image_size);
    img.src = './' + this.img_dir + '/' + this.img_date + this.PCE[j] + '.png';
    img.style.transform = "rotate(180deg)";
    img.id = "S_" + this.PCE[j];
    div.appendChild(img);
    //this.div_piece.appendChild(div);

    i = 5;
    top = i * (this.blank_size + this.border_width);
    left = (0 + j) * (this.blank_size + this.border_width);

    div = this.get_blank(i, (0 + j));

    style = "";
    style = "";
    style += "top :" + top + "px;";
    style += "left:" + left + "px;";
    div.setAttribute("style", style);

    img = new Image(this.image_size, this.image_size);
    img.src = './' + this.img_dir + '/' + this.img_date + this.PCE[j] + '.png';
    img.id = "F_" + this.PCE[j];
    div.appendChild(img);
    //this.div_piece.appendChild(div);
  }
  //jade
  var top = 1 * (this.blank_size + this.border_width);
  var left = 5 * (this.blank_size + this.border_width);
  var div = this.get_blank(1, 5);

  var style = "";
  style = "";
  style += "top :" + top + "px;";
  style += "left:" + left + "px;";
  div.setAttribute("style", style);

  let img = new Image(this.image_size, this.image_size);
  img.src = './' + this.img_dir + '/' + this.img_date + this.PCE[1] + '.png';
  img.style.transform = "rotate(180deg)";
  img.id = "S_" + this.PCE[1];
  div.appendChild(img);
  //this.div_piece.appendChild(div);

  //king
  top = 5 * (this.blank_size + this.border_width);
  left = 1 * (this.blank_size + this.border_width);
  div = this.get_blank(5, 1);

  style = "";
  style = "";
  style += "top :" + top + "px;";
  style += "left:" + left + "px;";
  div.setAttribute("style", style);

  img = new Image(this.image_size, this.image_size);
  img.src = './' + this.img_dir + '/' + this.img_date + this.PCE[0] + '.png';
  img.id = "F_" + this.PCE[0];
  div.appendChild(img);
  //this.div_piece.appendChild(div);

  //pawn
  top = 2 * (this.blank_size + this.border_width);
  left = 5 * (this.blank_size + this.border_width);
  div = this.get_blank(2, 5);

  style = "";
  style = "";
  style += "top :" + top + "px;";
  style += "left:" + left + "px;";
  div.setAttribute("style", style);

  img = new Image(this.image_size, this.image_size);
  img.src = './' + this.img_dir + '/' + this.img_date + this.PCE[6] + '.png';
  img.style.transform = "rotate(180deg)";
  img.id = "S_" + this.PCE[6];
  div.appendChild(img);
  //this.div_piece.appendChild(div);

  top = 4 * (this.blank_size + this.border_width);
  left = 1 * (this.blank_size + this.border_width);
  div = this.get_blank(4, 1);

  style = "";
  style = "";
  style += "top :" + top + "px;";
  style += "left:" + left + "px;";
  div.setAttribute("style", style);

  img = new Image(this.image_size, this.image_size);
  img.src = './' + this.img_dir + '/' + this.img_date + this.PCE[6] + '.png';
  img.id = "F_" + this.PCE[6];
  div.appendChild(img);

  /*// test img
  var top = 3 * (this.blank_size + this.border_width);
  var left = 3 * (this.blank_size + this.border_width);
  var div = this.get_blank(3, 3);

  var style = "";
  style = "";
  style += "top :" + top + "px;";
  style += "left:" + left+ "px;";
  div.setAttribute("style", style);

  img = new Image(this.image_size, this.image_size);
  img.src = './' + this.img_dir + '/' + this.img_date + this.PCE[5] + '.png';
  img.style.transform = "rotate(180deg)";
  img.id = "F_" + this.PCE[5];
  div.appendChild(img);
  */

  //this.div_piece.appendChild(div);

  //console.log("piece_init end");
}

MSBoard.prototype.screen_init = function () {

  var style = "";
  style = "";
  style += "top :" + (1 * (this.blank_size + this.border_width) + this.boundary_top) + "px;"
  style += "left:" + (3 * (this.blank_size + this.border_width) + this.boundary_left) + "px;"
  this.div_board.setAttribute("style", style);

  style = "";
  style += "top :" + (1 * (this.blank_size + this.border_width) + this.piece_bound_top + this.boundary_top) + "px;"
  style += "left:" + (3 * (this.blank_size + this.border_width) + this.piece_bound_left + this.boundary_left) + "px;"
  this.div_piece.setAttribute("style", style);

  style = "";
  style += "top :" + ((0 * (this.die_blank_size + this.border_width)) + (2.2 * (this.blank_size + this.border_width)) + this.boundary_top) + "px;"
  style += "left:" + (0.5 * (this.blank_size + this.border_width) + this.boundary_left) + "px;"
  //this.div_red_died.setAttribute("style", style);
  this.div_black_died.setAttribute("style", style);

  style = "";
  style += "top :" + ((6 * (this.die_blank_size + this.border_width)) + (2.2 * (this.blank_size + this.border_width)) + this.boundary_top) + "px;"
  style += "left:" + (0.5 * (this.blank_size + this.border_width) + this.boundary_left) + "px;"
  this.div_red_died.setAttribute("style", style);
  //this.div_black_died.setAttribute("style", style);

  style = "";
  style += "top :" + (1 * (this.blank_size + this.border_width) + this.boundary_top) + "px;"
  style += "left:" + (13 * (this.blank_size + this.border_width) + this.boundary_left) + "px;"
  this.div_movelist.setAttribute("style", style);

}

MSBoard.prototype.red_died_init = function () {

  //opponet
  var div = document.createElement('div');
  let img = new Image(280, 640);
  if (this.clr == 0) {
    //left
    img.src = './' + this.img_dir + '/' + this.img_date + 'sec.png';
  } else {
    //right 
    img.src = './' + this.img_dir + '/' + this.img_date + 'fir.png';
  }
  div.append(img);

  let leftr = (0.5 * (this.blank_size + this.border_width) + this.boundary_left);
  var style = "";
  style += "top :" + ((2.2 * (this.blank_size + this.border_width)) + this.boundary_top) + (3 * (this.blank_size + this.border_width) + this.boundary_left) + "px;"
  style += "left:" + leftr + "px;"
  this.div_red_died.setAttribute("style", style);


  for (let i = 1; i <= 5; i++) {
    let bl1 = this.new_div_shape("blank", i * (this.blank_size + this.border_width) + 20, (leftr + 10), this.blank_size, this.blank_size, "blank");
    let bl2 = this.new_div_shape("blank", i * (this.blank_size + this.border_width) + 20, (leftr + (this.blank_size + this.border_width) + 10), this.blank_size, this.blank_size, "blank");

    this.div_red_died.appendChild(bl1);
    this.div_red_died.appendChild(bl2);
  }
  this.div_red_died.appendChild(div);
}

MSBoard.prototype.black_died_init = function () {
  //my
  var div = document.createElement('div');

  let img = new Image(280, 640);
  if (this.clr == 0) {
    //left
    img.src = './' + this.img_dir + '/' + this.img_date + 'fir.png';
  } else {
    //right 
    img.src = './' + this.img_dir + '/' + this.img_date + 'sec.png';
  }
  div.append(img);

  let leftr = (0.5 * (this.blank_size + this.border_width) + this.boundary_left + 9 * (this.blank_size + this.border_width));
  let leftb = (0.5 * (this.blank_size + this.border_width) + this.boundary_left);
  var style = "";
  style += "top :" + ((1.4 * (this.blank_size + this.border_width)) + this.boundary_top) + "px;";
  style += "left:" + leftr + "px;";
  this.div_black_died.setAttribute("style", style);

  for (let i = 1; i <= 5; i++) {
    let bl1 = this.new_div_shape("blank", i * (this.blank_size + this.border_width) + 20, (leftb + 10), this.blank_size, this.blank_size, "");
    let bl2 = this.new_div_shape("blank", i * (this.blank_size + this.border_width) + 20, (leftb + (this.blank_size + this.border_width) + 10), this.blank_size, this.blank_size, "blank");

    this.div_black_died.appendChild(bl1);
    this.div_black_died.appendChild(bl2);
  }
  this.div_black_died.appendChild(div);
}

MSBoard.prototype.set_status = function (status) {

}

MSBoard.prototype.set_board = function (board) {
}

MSBoard.prototype.shape_mouse_click = function () {
  //console.log(this, this.firstChild);
  if (this.firstChild == null) { return; }
  var arr = new Array();
  arr = board.get_rowcol(this.id);
  var img_id = this.firstChild.id.split('_');
  let div_blank = null;
  let x = parseInt(arr[0]);
  let y = arr[1].charCodeAt() - 'a'.charCodeAt() + 1;

  /*ex:
   *  arr=>['5', 'd']
   *  img=>['F', 'Pawn']
  */
  console.log(arr, img_id, board.chose_pos);

  if (board.chose_pos == null) {
    board.chose_pos = this.id;
    board.chose_FS = img_id[0];
    board.set_blank(x, y, img_id[0], img_id[1]);
  } else if (board.chose_pos != null && board.chose_FS == img_id[0]) {
    board.remove_div("rect");
    board.set_blank(x, y, img_id[0], img_id[1]);
  } else if (board.chose_pos != null && board.chose_FS != img_id[0] && img_id[0] != "") {
    //if( this.firstElementChild.firstElementChild.className == "rect" ){
    console.log(this);
    //}
  } else {

  }
}


MSBoard.prototype.get_blank = function (row, col) {
  var cols = 6 - col;
  var rows = String.fromCharCode(this.EN_NUM[row].charCodeAt());;
  s = cols + "_" + rows;
  return document.getElementById(cols + "_" + rows);
}

MSBoard.prototype.get_rowcol = function (id) {
  var arr = new Array();
  arr = id.split('_');
  return arr;
}
MSBoard.prototype.set_blank = function (x, y, FS, piece) {

  switch (piece) {
    case "King":
    case "Jade":
      for (let i = -1; i < 2; i++) {
        if (y + i < 1 || y + i > 5) { continue; }
        for (let j = -1; j < 2; j++) {
          if ((6 - x + j) < 1 || (6 - x + j) > 5) { continue; }
          if (i == 0 && j == 0) { continue; }
          var div = create_div(85, 85, 0, 0);
          div_blank = board.get_blank(y + i, (6 - x + j));
          if (div_blank.firstChild != null) {
            if (div_blank.firstChild.id.split('_')[0] == FS) {
              continue;
            }
          }
          div_blank.appendChild(div);
        }
      }
      break;
    case "Gold":
      if (FS == 'S') {
        for (let i = -1; i < 2; i++) {
          if (y + i < 1 || y + i > 5) { continue; }
          for (let j = -1; j < 2; j++) {
            if ((6 - x + j) < 1 || (6 - x + j) > 5) { continue; }
            if (i == 0 && j == 0) { continue; }
            if (i == -1 && j != 0) { continue; }
            var div = create_div(85, 85, 0, 0);
            div_blank = board.get_blank(y + i, (6 - x + j));
            if (div_blank.firstChild != null) {
              if (div_blank.firstChild.id.split('_')[0] == FS) {
                continue;
              }
            }
            div_blank.appendChild(div);
          }
        }
      } else {
        for (let i = -1; i < 2; i++) {
          if (y + i < 1 || y + i > 5) { continue; }
          for (let j = -1; j < 2; j++) {
            if ((6 - x + j) < 1 || (6 - x + j) > 5) { continue; }
            if (i == 0 && j == 0) { continue; }
            if (i == 1 && j != 0) { continue; }
            var div = create_div(85, 85, 0, 0);
            div_blank = board.get_blank(y + i, (6 - x + j));
            if (div_blank.firstChild != null) {
              if (div_blank.firstChild.id.split('_')[0] == FS) {
                continue;
              }
            }
            div_blank.appendChild(div);
          }
        }
      }
      break;
    case "Silver":
      if (FS == 'S') {
        for (let i = -1; i < 2; i++) {
          if (y + i < 1 || y + i > 5) { continue; }
          for (let j = -1; j < 2; j++) {
            if ((6 - x + j) < 1 || (6 - x + j) > 5) { continue; }
            if (i == 0) { continue; }
            if (i == -1 && j == 0) { continue; }
            var div = create_div(85, 85, 0, 0);
            div_blank = board.get_blank(y + i, (6 - x + j));
            if (div_blank.firstChild != null) {
              if (div_blank.firstChild.id.split('_')[0] == FS) {
                continue;
              }
            }
            div_blank.appendChild(div);
          }
        }
      } else {
        for (let i = -1; i < 2; i++) {
          if (y + i < 1 || y + i > 5) { continue; }
          for (let j = -1; j < 2; j++) {
            if ((6 - x + j) < 1 || (6 - x + j) > 5) { continue; }
            if (i == 0) { continue; }
            if (i == 1 && j == 0) { continue; }
            var div = create_div(85, 85, 0, 0);
            div_blank = board.get_blank(y + i, (6 - x + j));
            if (div_blank.firstChild != null) {
              if (div_blank.firstChild.id.split('_')[0] == FS) {
                continue;
              }
            }
            div_blank.appendChild(div);
          }
        }
      }
      break;
    case "Bishop":
      //角
      console.log("x y:", x, y);
      for (let i = -4; i <= 4; i++) {
        console.log("i:", i, y + i, (6 - x + i));
        if (y + i < 1 || y + i > 5) { continue; }
        if ((6 - x + i) < 1 || (6 - x + i) > 5) { continue; }
        if (i == 0) { continue; }
        var div = create_div(85, 85, 0, 0);
        div_blank = board.get_blank(y + i, (6 - x + i));
        div_blank.appendChild(div);
        let j = -i;
        if ((6 - x + j) < 1 || (6 - x + j) > 5) { continue; }
        var div = create_div(85, 85, 0, 0);
        div_blank = board.get_blank(y + i, (6 - x + j));
        if (div_blank.firstChild != null) {
          if (div_blank.firstChild.id.split('_')[0] == FS) {
            continue;
          }
        }
        div_blank.appendChild(div);
      }
      break;
    case "Rook":
      //row
      let negative = 1;
      let positive = 1;
      for (let i = 1; i < 6; i++) {
        if (negative && y - i > 0) {
          var div = create_div(85, 85, 0, 0);
          div_blank = board.get_blank(y - i, (6 - x));
          if (div_blank.firstChild != null) {
            console.log(div_blank.firstChild.id, FS)
            if (div_blank.firstChild.id.split('_')[0] == FS) {
              negative = 0;
              break;
            } else {
              negative = 0;
              div_blank.appendChild(div);
              break;
            }
          }
          div_blank.appendChild(div);
        }
        if (positive && y + i < 6) {
          var div = create_div(85, 85, 0, 0);
          div_blank = board.get_blank(y + i, (6 - x));
          if (div_blank.firstChild != null) {
            console.log(div_blank.firstChild.id, FS)
            if (div_blank.firstChild.id.split('_')[0] == FS) {
              positive = 0;
              break;
            } else {
              positive = 0;
              div_blank.appendChild(div);
              break;
            }
          }
          div_blank.appendChild(div);
        }
      }
      //col
      negative = 1;
      positive = 1;
      for (let i = 1; i < 6; i++) {
        if (negative && (6 - x - i) > 0) {
          var div = create_div(85, 85, 0, 0);
          div_blank = board.get_blank(y, (6 - x - i));
          if (div_blank.firstChild != null) {
            if (div_blank.firstChild.id.split('_')[0] == FS) {
              negative = 0;
              break;
            } else {
              negative = 0;
              div_blank.appendChild(div);
              break;
            }
          }
          div_blank.appendChild(div);
        }
        if (positive && (6 - x + i) < 6) {
          var div = create_div(85, 85, 0, 0);
          div_blank = board.get_blank(y, (6 - x + i));
          if (div_blank.firstChild != null) {
            if (div_blank.firstChild.id.split('_')[0] == FS) {
              positive = 0;
              break;
            } else {
              positive = 0;
              div_blank.appendChild(div);
              break;
            }
          }
          div_blank.appendChild(div);
        }
      }
      break;
    case "Pawn":
      if (FS == 'S') {
        var div = create_div(85, 85, 0, 0);
        div_blank = board.get_blank(y + 1, (6 - x));
        if (div_blank.firstChild != null) {
          if (div_blank.firstChild.id.split('_')[0] == FS) {
            break;
          }
        }
        div_blank.appendChild(div);
      } else {
        var div = create_div(85, 85, 0, 0);
        div_blank = board.get_blank(y - 1, (6 - x));
        if (div_blank.firstChild != null) {
          if (div_blank.firstChild.id.split('_')[0] == FS) {
            break;
          }
        }
        div_blank.appendChild(div);
      }
      break;
  }
}
console.log("js loading...");